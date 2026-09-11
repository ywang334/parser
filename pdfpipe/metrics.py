"""Ground-truth-only table metrics. Parser agreement is deliberately excluded."""
import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass

from apted import APTED, Config
from distance import levenshtein
from lxml import html as lhtml

from .tables import iou, render

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class Node:
    tag: str
    attrs: tuple = ()
    text: str = ''
    children: tuple = ()


class TEDSConfig(Config):
    def __init__(self, structure_only=False):
        self.structure_only = structure_only

    def rename(self, a, b):
        if a.tag != b.tag or a.attrs != b.attrs:
            return 1
        if self.structure_only or a.tag not in ('td', 'th'):
            return 0
        maximum = max(len(a.text), len(b.text))
        return levenshtein(a.text, b.text) / maximum if maximum else 0

    def children(self, node):
        return node.children


def _tree(source):
    roots = lhtml.fromstring(source).xpath('.//table|self::table')
    if not roots:
        raise ValueError('ground truth/prediction has no table')
    allowed = {'table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th'}
    def build(elem):
        children = tuple(build(c) for c in elem if c.tag in allowed)
        attrs = ()
        if elem.tag in ('td', 'th'):
            attrs = (int(elem.get('rowspan', '1')), int(elem.get('colspan', '1')))
        text = re.sub(r'\s+', ' ', ' '.join(elem.itertext())).strip() if elem.tag in ('td', 'th') else ''
        return Node(elem.tag, attrs, text, children)
    return build(roots[0])


def _size(node):
    return 1 + sum(_size(c) for c in node.children)


def teds(true_html, pred_html, structure_only=False):
    true_tree, pred_tree = _tree(true_html), _tree(pred_html)
    denominator = max(_size(true_tree), _size(pred_tree))
    distance = APTED(true_tree, pred_tree, TEDSConfig(structure_only)).compute_edit_distance()
    return max(0.0, 1.0 - distance / denominator)


def _match(gt, predictions):
    matched, used = [], set()
    for truth in gt:
        explicit = truth.get('prediction_id')
        candidates = [(j, p) for j, p in enumerate(predictions) if j not in used]
        if explicit:
            candidates = [(j, p) for j, p in candidates if p.get('id') == explicit]
        else:
            candidates = [(j, p) for j, p in candidates if p.get('page') == truth.get('page')]
            candidates.sort(key=lambda x: iou(truth.get('bbox'), x[1].get('bbox')), reverse=True)
            if candidates and iou(truth.get('bbox'), candidates[0][1].get('bbox')) < .25:
                candidates = []
        if candidates:
            used.add(candidates[0][0]); matched.append((truth, candidates[0][1]))
        else:
            matched.append((truth, None))
    return matched, [p for j, p in enumerate(predictions) if j not in used]


def evaluate(ground_truth, prediction):
    official=ROOT/'vendor/sources/table-transformer/src'
    if not official.exists(): official=ROOT/'vendor/table-transformer-upstream/src'
    sys.path.insert(0, str(official))
    import numpy as np
    import grits
    # Current PyMuPDF Rect rejects numpy.ndarray while the official GriTS code
    # passes grid entries as ndarrays. Convert only the boundary type.
    original_rect=grits.Rect
    def compatible_rect(*args, **kwargs):
        if len(args) == 1 and hasattr(args[0], 'tolist'):
            args=(args[0].tolist(),)
        return original_rect(*args, **kwargs)
    grits.Rect=compatible_rect
    def compatible_iou(a, b):
        a=a.tolist() if hasattr(a, 'tolist') else a
        b=b.tolist() if hasattr(b, 'tolist') else b
        intersection=original_rect(a).intersect(original_rect(b))
        union=original_rect(a).include_rect(original_rect(b))
        return intersection.get_area()/union.get_area() if union.get_area() > 0 else 0
    grits.iou=compatible_iou
    cells_to_grid,grits_from_html,grits_loc=grits.cells_to_grid,grits.grits_from_html,grits.grits_loc
    pairs, false_positives = _match(ground_truth['tables'], prediction.get('tables', []))
    rows = []
    for truth, pred in pairs:
        if pred is None:
            rows.append({'ground_truth_id': truth['id'], 'prediction_id': None,
                         'teds': 0., 'teds_s': 0., 'grits_top': 0., 'grits_con': 0.,
                         'grits_loc': 0. if truth.get('cells') else None, 'status': 'missed'})
            continue
        pred_html = pred.get('html') or render(pred)
        gt_html = truth['html']
        grits = grits_from_html(gt_html, pred_html)
        loc = None
        if truth.get('cells') and pred.get('cells') and all(c.get('bbox') for c in truth['cells'] + pred['cells']):
            def official_cells(cells):
                return [{'row_nums': list(range(c['row'], c['row']+c.get('rowspan', 1))),
                         'column_nums': list(range(c['col'], c['col']+c.get('colspan', 1))),
                         'bbox': c['bbox']} for c in cells]
            value = grits_loc(np.array(cells_to_grid(official_cells(truth['cells']))),
                              np.array(cells_to_grid(official_cells(pred['cells']))))
            loc = float(value[0])
        rows.append({'ground_truth_id': truth['id'], 'prediction_id': pred['id'],
                     'teds': teds(gt_html, pred_html), 'teds_s': teds(gt_html, pred_html, True),
                     'grits_top': float(grits['grits_top']), 'grits_con': float(grits['grits_con']),
                     'grits_loc': loc,
                     'status': 'matched'})
    fields = ('teds', 'teds_s', 'grits_top', 'grits_con', 'grits_loc')
    count = len(rows)
    aggregate = {field: (sum(row[field] for row in rows if row[field] is not None) /
                         sum(row[field] is not None for row in rows)) if any(row[field] is not None for row in rows) else None
                 for field in fields}
    return {'metric_basis': 'provided_ground_truth', 'ground_truth_tables': count,
            'false_positive_tables': [p['id'] for p in false_positives], 'aggregate': aggregate, 'tables': rows,
            'notes': ['TEDS-S ignores cell text.', 'GriTS_Loc is reported only when both sides have cell boxes.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('ground_truth');p.add_argument('prediction');p.add_argument('--out',required=True)
    a=p.parse_args();gt=json.loads(pathlib.Path(a.ground_truth).read_text());pred=json.loads(pathlib.Path(a.prediction).read_text())
    pathlib.Path(a.out).write_text(json.dumps(evaluate(gt,pred),ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
