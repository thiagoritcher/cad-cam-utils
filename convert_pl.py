import argparse
from collections import defaultdict
from pathlib import Path
import ezdxf
from ezdxf import edgeminer, edgesmith


def converter_por_layer_mm(
    arquivo_entrada: str,
    layers_ignorados: list[str],
    tolerancia_mm: float = 0.01,
):
    caminho_entrada = Path(arquivo_entrada)

    if not caminho_entrada.exists():
        print(f"Erro: Arquivo '{caminho_entrada}' não encontrado.")
        return

    caminho_saida = caminho_entrada.with_name(
        f"{caminho_entrada.stem}-pl{caminho_entrada.suffix}"
    )

    doc = ezdxf.readfile(caminho_entrada)
    doc.header["$INSUNITS"] = 4  # Define unidade para Milímetros (4 = mm)
    msp = doc.modelspace()

    ignorados_set = {layer.lower() for layer in layers_ignorados}

    # 1. Agrupa todas as linhas e arcos por Layer (pulando os ignorados)
    entidades_por_layer = defaultdict(list)
    for ent in msp:
        if ent.dxftype() in ("LINE", "ARC"):
            nome_layer = ent.dxf.layer
            if nome_layer.lower() not in ignorados_set:
                entidades_por_layer[nome_layer].append(ent)

    if not entidades_por_layer:
        print(
            "Nenhuma linha ou arco elegível encontrado após filtrar os layers."
        )
        return

    entidades_para_remover = set()

    # 2. Processa cada Layer válido de forma isolada
    for nome_layer, entidades in entidades_por_layer.items():
        if len(entidades) < 2:
            continue

        # Converte entidades DXF em arestas 2D
        edges = list(
            edgesmith.edges_from_entities_2d(entidades, gap_tol=tolerancia_mm)
        )

        # Passa a lista 'edges' e a tolerância diretamente para encontrar as cadeias
        cadeias = edgeminer.find_all_sequential_chains(
            edges, gap_tol=tolerancia_mm
        )

        for cadeia in cadeias:
            # Ignora segmentos isolados
            if len(cadeia) < 2:
                continue

            # Converte a cadeia diretamente em LWPOLYLINE preservando os arcos (bulge)
            nova_poly = edgesmith.lwpolyline_from_chain(
                cadeia, dxfattribs={"layer": nome_layer}
            )
            msp.add_entity(nova_poly)

            # Marca as entidades originais (armazenadas no payload) para remoção
            for edge in cadeia:
                if edge.payload is not None:
                    entidades_para_remover.add(edge.payload)

    # 3. Remove as linhas e arcos originais que foram unidos
    for ent in entidades_para_remover:
        msp.delete_entity(ent)

    doc.saveas(caminho_saida)
    print(f"Conversão concluída com sucesso!")
    print(f"Arquivo gerado: {caminho_saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte linhas e arcos conectados em polylines por layer (em mm)."
    )
    parser.add_argument(
        "arquivo",
        help="Caminho do arquivo DXF de entrada (ex: desenho.dxf)",
    )
    parser.add_argument(
        "-i",
        "--ignorar",
        nargs="*",
        default=["DEFPOINTS", "COTAS", "EIXOS", "CONSTRUCAO"],
        help="Lista de layers para ignorar (separados por espaço)",
    )
    parser.add_argument(
        "-t",
        "--tolerancia",
        type=float,
        default=0.01,
        help="Tolerância de união em mm (padrão: 0.01)",
    )

    args = parser.parse_args()
    converter_por_layer_mm(args.arquivo, args.ignorar, args.tolerancia)

