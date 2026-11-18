import argparse
import math
from pathlib import Path
import ezdxf
from ezdxf.math import Vec2
import numpy as np


def fit_circle(points):
    """Ajusta um círculo (centro e raio) a uma lista de pontos 2D usando mínimos quadrados."""
    pts = np.array(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m

    Suv = np.sum(u * v)
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0

    if np.linalg.cond(A) > 1e10:
        return None

    uc, vc = np.linalg.solve(A, B)
    xc = x_m + uc
    yc = y_m + vc
    R = float(np.mean(np.sqrt((x - xc) ** 2 + (y - yc) ** 2)))
    return (float(xc), float(yc)), R


def validate_vertices_radial_error(pts, center, radius):
    """Calcula o erro máximo radial nos vértices reais da poligonal."""
    cx, cy = center
    max_err = 0.0
    for px, py in pts:
        radial_err = abs(math.hypot(px - cx, py - cy) - radius)
        if radial_err > max_err:
            max_err = radial_err
    return max_err


def normalize_angle(angle_deg):
    """Normaliza um ângulo em graus para o intervalo [0, 360)."""
    return angle_deg % 360.0


def angle_in_ccw_arc(angle_deg, start_deg, end_deg, tol_deg=1e-3):
    """Verifica se um ângulo está dentro do arco anti-horário de start_deg até end_deg."""
    a = normalize_angle(angle_deg - start_deg)
    span = normalize_angle(end_deg - start_deg)
    return a <= span + tol_deg


def build_exact_arc_params(sub_pts, center):
    """Calcula start_angle e end_angle reais em graus (sentido anti-horário do DXF)."""
    cx, cy = center
    p_start = sub_pts[0]
    p_end = sub_pts[-1]

    ang_start = normalize_angle(
        math.degrees(math.atan2(p_start[1] - cy, p_start[0] - cx))
    )
    ang_end = normalize_angle(
        math.degrees(math.atan2(p_end[1] - cy, p_end[0] - cx))
    )

    mid_pts = sub_pts[1:-1]
    ccw_hits = 0
    for p in mid_pts:
        ang_p = normalize_angle(math.degrees(math.atan2(p[1] - cy, p[0] - cx)))
        if angle_in_ccw_arc(ang_p, ang_start, ang_end):
            ccw_hits += 1

    if ccw_hits >= len(mid_pts) / 2.0:
        final_start, final_end = ang_start, ang_end
    else:
        final_start, final_end = ang_end, ang_start

    for p in mid_pts:
        ang_p = normalize_angle(math.degrees(math.atan2(p[1] - cy, p[0] - cx)))
        if not angle_in_ccw_arc(ang_p, final_start, final_end, tol_deg=1.0):
            return None

    return final_start, final_end


def validate_arc_error(sub_pts, center, radius, start_angle, end_angle):
    """Calcula o erro máximo nos vértices e nas extremidades do arco DXF."""
    cx, cy = center
    max_err = validate_vertices_radial_error(sub_pts, center, radius)

    p_arc_start = (
        cx + radius * math.cos(math.radians(start_angle)),
        cy + radius * math.sin(math.radians(start_angle)),
    )
    p_arc_end = (
        cx + radius * math.cos(math.radians(end_angle)),
        cy + radius * math.sin(math.radians(end_angle)),
    )

    d1 = math.hypot(
        p_arc_start[0] - sub_pts[0][0], p_arc_start[1] - sub_pts[0][1]
    ) + math.hypot(
        p_arc_end[0] - sub_pts[-1][0], p_arc_end[1] - sub_pts[-1][1]
    )
    d2 = math.hypot(
        p_arc_start[0] - sub_pts[-1][0], p_arc_start[1] - sub_pts[-1][1]
    ) + math.hypot(
        p_arc_end[0] - sub_pts[0][0], p_arc_end[1] - sub_pts[0][1]
    )
    endpoint_err = min(d1, d2) / 2.0

    return max(max_err, endpoint_err)


def extract_chains_by_layer(msp, ignore_layers=None, chain_tol=1e-3):
    """Agrupa entidades LINE, LWPOLYLINE, POLYLINE, ARC e SPLINE conectadas por layer."""
    ignore_set = (
        {layer.upper() for layer in ignore_layers} if ignore_layers else set()
    )

    segments_by_layer = {}
    to_delete = []
    entity_counts = {}

    for e in msp:
        etype = e.dxftype()
        entity_counts[etype] = entity_counts.get(etype, 0) + 1

        layer = e.dxf.layer
        if layer.upper() in ignore_set:
            continue

        if layer not in segments_by_layer:
            segments_by_layer[layer] = []

        if etype == "LINE":
            p1 = (float(e.dxf.start.x), float(e.dxf.start.y))
            p2 = (float(e.dxf.end.x), float(e.dxf.end.y))
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 1e-6:
                segments_by_layer[layer].append([p1, p2])
                to_delete.append(e)

        elif etype == "LWPOLYLINE":
            pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
            if e.closed and len(pts) > 1:
                pts.append(pts[0])
            for i in range(len(pts) - 1):
                if (
                    math.hypot(
                        pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
                    )
                    > 1e-6
                ):
                    segments_by_layer[layer].append([pts[i], pts[i + 1]])
            to_delete.append(e)

        elif etype == "POLYLINE":
            pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
            if e.is_closed and len(pts) > 1:
                pts.append(pts[0])
            for i in range(len(pts) - 1):
                if (
                    math.hypot(
                        pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
                    )
                    > 1e-6
                ):
                    segments_by_layer[layer].append([pts[i], pts[i + 1]])
            to_delete.append(e)

        elif etype == "SPLINE":
            pts = [(float(p.x), float(p.y)) for p in e.flattening(distance=0.01)]
            for i in range(len(pts) - 1):
                if (
                    math.hypot(
                        pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
                    )
                    > 1e-6
                ):
                    segments_by_layer[layer].append([pts[i], pts[i + 1]])
            to_delete.append(e)

        elif etype == "ARC":
            pts = [(float(p.x), float(p.y)) for p in e.flattening(sagitta=0.01)]
            for i in range(len(pts) - 1):
                if (
                    math.hypot(
                        pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
                    )
                    > 1e-6
                ):
                    segments_by_layer[layer].append([pts[i], pts[i + 1]])
            to_delete.append(e)

    for e in to_delete:
        msp.delete_entity(e)

    chains_by_layer = {}
    for layer, segments in segments_by_layer.items():
        chains = []
        while segments:
            chain = segments.pop(0)
            changed = True
            while changed:
                changed = False
                for i, seg in enumerate(segments):
                    if math.hypot(chain[-1][0] - seg[0][0], chain[-1][1] - seg[0][1]) <= chain_tol:
                        chain.append(seg[1])
                        segments.pop(i)
                        changed = True
                        break
                    elif math.hypot(chain[-1][0] - seg[1][0], chain[-1][1] - seg[1][1]) <= chain_tol:
                        chain.append(seg[0])
                        segments.pop(i)
                        changed = True
                        break
                    elif math.hypot(chain[0][0] - seg[1][0], chain[0][1] - seg[1][1]) <= chain_tol:
                        chain.insert(0, seg[0])
                        segments.pop(i)
                        changed = True
                        break
                    elif math.hypot(chain[0][0] - seg[0][0], chain[0][1] - seg[0][1]) <= chain_tol:
                        chain.insert(0, seg[1])
                        segments.pop(i)
                        changed = True
                        break
            chains.append(chain)
        chains_by_layer[layer] = chains

    return chains_by_layer, entity_counts


def has_uniform_segments(pts, max_ratio=3.5):
    """Impede que segmentos curtos de concordância sejam misturados com linhas retas longas."""
    lengths = []
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        if d > 1e-6:
            lengths.append(d)
    if len(lengths) < 2:
        return True
    # Ignora eventual micro-segmento de fechamento/transição na ponta se os demais forem uniformes
    core = lengths[1:-1] if len(lengths) >= 4 else lengths
    if min(core) <= 0:
        return False
    return (max(core) / min(core)) <= max_ratio


def has_consistent_turning(pts, min_turn_deg=2.0, max_turn_deg=60.0):
    """Garante curvatura contínua no mesmo sentido, bloqueando linhas retas adjacentes."""
    signs = []
    for i in range(1, len(pts) - 1):
        v1 = Vec2(pts[i]) - Vec2(pts[i - 1])
        v2 = Vec2(pts[i + 1]) - Vec2(pts[i])
        if v1.magnitude == 0 or v2.magnitude == 0:
            continue
        ang_deg = math.degrees(v1.angle_between(v2))
        abs_deg = abs(ang_deg)
        if abs_deg < min_turn_deg or abs_deg > max_turn_deg:
            return False
        signs.append(1 if ang_deg > 0 else -1)
    if not signs:
        return False
    return all(s == signs[0] for s in signs)


def process_chain(
    msp, pts, layer, tolerance_mm=0.05, min_segments=3, close_tol=1e-2
):
    """Converte contornos fechados circulares em CIRCLE e trechos curvos em ARC."""
    n = len(pts)
    attribs = {"layer": layer}
    stats = {"circles": 0, "arcs": 0, "lines": 0}

    # 1. VERIFICAÇÃO DE CÍRCULO FECHADO COMPLETO (mínimo 6 lados = 7 pontos)
    is_closed = (
        n >= 7
        and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
        <= close_tol
    )

    if is_closed:
        unique_pts = pts[:-1]
        closed_check_pts = pts + [pts[1]]
        if has_consistent_turning(closed_check_pts, min_turn_deg=2.0, max_turn_deg=60.0) and has_uniform_segments(pts):
            res = fit_circle(unique_pts)
            if res is not None:
                center, radius = res
                if tolerance_mm < radius <= 100000.0:
                    err = validate_vertices_radial_error(unique_pts, center, radius)
                    if err <= tolerance_mm:
                        msp.add_circle(
                            center=center, radius=radius, dxfattribs=attribs
                        )
                        stats["circles"] += 1
                        return stats

    # 2. CONVERSÃO EM ARCOS E LINHAS PARA CONTORNOS ABERTOS OU MISTOS
    i = 0
    while i < n - 1:
        best_j = -1
        best_arc = None

        for j in range(i + min_segments, n + 1):
            sub_pts = pts[i:j]

            if not has_consistent_turning(sub_pts, min_turn_deg=2.0, max_turn_deg=60.0):
                break
            if not has_uniform_segments(sub_pts):
                break

            res = fit_circle(sub_pts)
            if res is None:
                break

            center, radius = res
            if radius > 100000.0 or radius < tolerance_mm:
                break

            arc_angles = build_exact_arc_params(sub_pts, center)
            if arc_angles is None:
                break

            start_angle, end_angle = arc_angles
            err = validate_arc_error(
                sub_pts, center, radius, start_angle, end_angle
            )

            if err <= tolerance_mm:
                best_j = j
                best_arc = (center, radius, start_angle, end_angle)
            else:
                break

        if best_j != -1:
            center, radius, start_angle, end_angle = best_arc
            msp.add_arc(
                center=center,
                radius=radius,
                start_angle=start_angle,
                end_angle=end_angle,
                dxfattribs=attribs,
            )
            stats["arcs"] += 1
            i = best_j - 1
        else:
            msp.add_line(pts[i], pts[i + 1], dxfattribs=attribs)
            stats["lines"] += 1
            i += 1

    return stats


def convert_dxf_lines_to_arcs(
    input_file,
    tolerance_mm=0.05,
    min_segments=3,
    ignore_layers=None,
):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    output_path = input_path.with_name(
        f"{input_path.stem}-arc{input_path.suffix}"
    )

    doc = ezdxf.readfile(input_path)
    doc.header["$INSUNITS"] = 4  # Milímetros
    msp = doc.modelspace()

    chain_tol = max(1e-3, tolerance_mm * 0.2)
    close_tol = max(1e-2, tolerance_mm * 0.5)

    chains_by_layer, entity_counts = extract_chains_by_layer(
        msp, ignore_layers=ignore_layers, chain_tol=chain_tol
    )

    total_stats = {"circles": 0, "arcs": 0, "lines": 0}
    total_chains = 0

    for layer, chains in chains_by_layer.items():
        total_chains += len(chains)
        for chain in chains:
            st = process_chain(
                msp,
                chain,
                layer=layer,
                tolerance_mm=tolerance_mm,
                min_segments=min_segments,
                close_tol=close_tol,
            )
            for k in total_stats:
                total_stats[k] += st[k]

    # Salva forçando a versão DXF AC1015 (AutoCAD 2000)
    doc.saveas(output_path, fmt="asc")
    # Reabre ou cria cópia limpa em AC1015 caso o arquivo original fosse R12 (AC1009)
    if doc.dxfversion != "AC1015":
        doc_ac1015 = ezdxf.new(dxfversion="AC1015")
        doc_ac1015.header["$INSUNITS"] = 4
        for layer in doc.layers:
            if layer.dxf.name not in doc_ac1015.layers:
                doc_ac1015.layers.add(
                    layer.dxf.name,
                    color=layer.dxf.color,
                    linetype=layer.dxf.linetype if layer.dxf.linetype in doc_ac1015.linetypes else "CONTINUOUS",
                )
        from ezdxf.addons import Importer
        importer = Importer(doc, doc_ac1015)
        importer.import_modelspace()
        importer.finalize()
        doc_ac1015.saveas(output_path)

    print(f"Entidades lidas no original: {entity_counts}")
    print(f"Cadeias contínuas montadas: {total_chains}")
    print(
        f"Resultado gerado -> Círculos: {total_stats['circles']} | "
        f"Arcos: {total_stats['arcs']} | Linhas: {total_stats['lines']}"
    )
    print(f"Versão DXF salva: AC1015 (AutoCAD 2000)")
    print(f"Arquivo salvo em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte segmentos de linha em arcos e círculos em arquivos DXF (mm / AC1015)."
    )
    parser.add_argument(
        "arquivo",
        help="Caminho do arquivo DXF de entrada (ex: peca.dxf)",
    )
    parser.add_argument(
        "-t",
        "--tolerancia",
        type=float,
        default=0.05,
        help="Tolerância máxima em mm (padrão: 0.05)",
    )
    parser.add_argument(
        "-m",
        "--min-segmentos",
        type=int,
        default=3,
        help="Mínimo de segmentos seguidos para gerar arco (padrão: 3)",
    )
    parser.add_argument(
        "-i",
        "--ignorar",
        nargs="*",
        default=["COTAS", "TEXTO", "DOBRA", "DEFPOINTS"],
        help="Lista de layers a ignorar separadas por espaço",
    )

    args = parser.parse_args()

    convert_dxf_lines_to_arcs(
        input_file=args.arquivo,
        tolerance_mm=args.tolerancia,
        min_segments=args.min_segmentos,
        ignore_layers=args.ignorar,
    )
