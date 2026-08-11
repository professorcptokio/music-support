"""Gera uma lead sheet (melodia + cifras) a partir dos arquivos da pasta input.

Para cada arquivo produz em output/:
  - <nome>.musicxml  partitura com melodia, cifras, armadura, formula de compasso
  - <nome>.mid       mesma coisa em MIDI, para conferir de ouvido se a analise bate

A analise harmonica vem inteira do analyze.py; aqui entra a transcricao da
melodia (pyin) e a diagramacao em compassos.

Uso:
    python partitura.py                    # processa tudo que houver em input/
    python partitura.py input/x.mp4        # processa apenas os arquivos indicados
    python partitura.py --subdiv 4         # resolucao de semicolcheia (padrao: colcheia)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import librosa
import numpy as np

from analyze import (
    EXTENSOES,
    INPUT_DIR,
    NOTAS,
    OUTPUT_DIR,
    QUALIDADES,
    Analise,
    analisar_arquivo,
    limites_compassos,
)

# Faixa onde a melodia e procurada. A CQT vai alem no agudo (ate NOTA_TETO) porque
# a saliencia precisa enxergar os harmonicos de cada altura candidata.
NOTA_MIN = "C3"
NOTA_MAX = "C6"
NOTA_TETO = "C8"

# Da coluna de saliencia, pega o bin mais agudo que chegue a esta fracao do pico:
# a melodia costuma ser a voz superior, e nao a mais forte (essa e o baixo).
LIMIAR_TOPO = 0.4

# Abaixo desta fracao da saliencia mediana o trecho e considerado silencio.
LIMIAR_SILENCIO = 0.25

# Fracao do slot que precisa ter altura definida para virar nota (senao, pausa).
LIMIAR_VOZ = 0.4

TICKS_POR_SEMINIMA = 480
PROGRAMA_MELODIA = 73  # flauta, so para destacar a melodia do acompanhamento
PROGRAMA_ACORDES = 0   # piano

NOTAS_SUSTENIDO = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTAS_BEMOL = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Tonica maior -> numero de acidentes na armadura.
ARMADURAS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5}

# Sufixo interno -> <kind> do MusicXML.
TIPOS_MUSICXML = {
    "": "major",
    "m": "minor",
    "7": "dominant",
    "m7": "minor-seventh",
    "maj7": "major-seventh",
    "sus4": "suspended-fourth",
    "dim": "diminished",
}

INTERVALOS = {sufixo: intervalos for sufixo, intervalos, _ in QUALIDADES}

FIGURAS = [("whole", 4.0), ("half", 2.0), ("quarter", 1.0),
           ("eighth", 0.5), ("16th", 0.25), ("32nd", 0.125)]


@dataclass
class Grupo:
    """Um trecho de melodia sem cortes internos: nota unica ou pausa."""

    inicio: int          # em slots, absoluto
    duracao: int         # em slots
    altura: int | None   # nota MIDI, ou None para pausa
    compasso: int        # indice do compasso a que pertence


# ------------------------------------------------------------------- melodia

def rastrear_melodia(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Altura da voz superior quadro a quadro. Devolve (tempos, notas MIDI ou -1).

    Um detector de f0 comum (pyin, por exemplo) trava no baixo quando o audio tem
    acompanhamento: e a componente mais periodica do mix. Aqui a busca e por
    saliencia harmonica, pegando o bin mais agudo que ainda esta bem sustentado,
    que e onde a melodia costuma estar.
    """
    y_harm, _ = librosa.effects.hpss(y)

    grave = float(librosa.note_to_hz(NOTA_MIN))
    n_bins = int(round(12 * np.log2(librosa.note_to_hz(NOTA_TETO) / grave)))
    cqt = np.abs(librosa.cqt(y_harm, sr=sr, fmin=grave, n_bins=n_bins, bins_per_octave=12))
    freqs = librosa.cqt_frequencies(n_bins, fmin=grave, bins_per_octave=12)
    saliencia = librosa.salience(cqt, freqs=freqs, harmonics=[1, 2, 3, 4],
                                 weights=[1.0, 0.5, 0.33, 0.25], fill_value=0)

    alcance = int(round(12 * np.log2(librosa.note_to_hz(NOTA_MAX) / grave)))
    faixa = saliencia[:alcance]
    picos = faixa.max(axis=0)
    positivos = picos[picos > 0]
    corte = LIMIAR_SILENCIO * float(np.median(positivos)) if len(positivos) else np.inf

    base = int(round(librosa.hz_to_midi(grave)))
    notas = np.full(faixa.shape[1], -1)
    for t in np.flatnonzero(picos > corte):
        coluna = faixa[:, t]
        candidatos = np.flatnonzero(coluna >= LIMIAR_TOPO * coluna.max())
        notas[t] = base + int(candidatos[-1])

    return librosa.times_like(faixa, sr=sr), notas


def extrair_melodia(a: Analise, subdiv: int) -> list[int | None]:
    """Encaixa a melodia rastreada na grade de slots (subdiv slots por batida)."""
    tempos, notas = rastrear_melodia(a.y, a.sr)

    slots: list[int | None] = []
    for i in range(len(a.acordes)):
        t0, t1 = a.tempos_batidas[i], a.tempos_batidas[i + 1]
        for s in range(subdiv):
            ini = t0 + (t1 - t0) * s / subdiv
            fim = t0 + (t1 - t0) * (s + 1) / subdiv
            janela = notas[(tempos >= ini) & (tempos < fim)]
            definidas = janela[janela >= 0]
            if len(janela) and len(definidas) >= LIMIAR_VOZ * len(janela):
                valores, repeticoes = np.unique(definidas, return_counts=True)
                slots.append(int(valores[np.argmax(repeticoes)]))
            else:
                slots.append(None)

    return corrigir_oitavas(slots)


def corrigir_oitavas(slots: list[int | None]) -> list[int | None]:
    """Conserta o erro classico do pyin: um slot solto uma oitava fora dos vizinhos."""
    corrigidos = list(slots)
    for i in range(1, len(slots) - 1):
        atual, antes, depois = slots[i], slots[i - 1], slots[i + 1]
        if atual is None or antes is None or depois is None:
            continue
        if antes == depois and abs(atual - antes) == 12:
            corrigidos[i] = antes
    return corrigidos


# ------------------------------------------------------------------ compassos

def agrupar(slots: list[int | None], compassos: list[tuple[int, int]],
            acordes: list[str], subdiv: int) -> list[Grupo]:
    """Quebra a melodia em grupos, cortando nas barras e nas trocas de acorde.

    Cortar na troca de acorde permite ancorar a cifra exatamente onde ela entra;
    quando a nota atravessa o corte, ela volta a se unir por ligadura.
    """
    cortes = {i * subdiv for i in range(1, len(acordes)) if acordes[i] != acordes[i - 1]}

    grupos: list[Grupo] = []
    for indice, (batida, n_batidas) in enumerate(compassos):
        s0 = batida * subdiv
        s1 = s0 + n_batidas * subdiv
        s = s0
        while s < s1:
            fim = s + 1
            while fim < s1 and fim not in cortes and slots[fim] == slots[s]:
                fim += 1
            grupos.append(Grupo(s, fim - s, slots[s], indice))
            s = fim
    return grupos


def decompor(duracao: int, divisoes: int) -> list[tuple[int, str, int]]:
    """Quebra uma duracao em figuras representaveis: (unidades, tipo, pontos)."""
    tabela: dict[int, tuple[str, int]] = {}
    for nome, fator in FIGURAS:
        for pontos, multiplicador in ((0, 1.0), (1, 1.5)):
            unidades = fator * multiplicador * divisoes
            if abs(unidades - round(unidades)) < 1e-9 and round(unidades) >= 1:
                tabela.setdefault(round(unidades), (nome, pontos))

    ordenadas = sorted(tabela.items(), reverse=True)
    partes, restante = [], duracao
    while restante > 0:
        for unidades, (nome, pontos) in ordenadas:
            if unidades <= restante:
                partes.append((unidades, nome, pontos))
                restante -= unidades
                break
        else:
            break
    return partes


# ------------------------------------------------------------------ musicxml

def nome_nota(pc: int, acidentes: int) -> str:
    return (NOTAS_SUSTENIDO if acidentes >= 0 else NOTAS_BEMOL)[pc % 12]


def separar_acorde(nome: str) -> tuple[int, str]:
    """'C#m7' -> (1, 'm7')"""
    corte = 2 if len(nome) > 1 and nome[1] == "#" else 1
    return NOTAS.index(nome[:corte]), nome[corte:]


def xml_harmonia(nome: str, acidentes: int) -> str:
    pc, sufixo = separar_acorde(nome)
    escrita = nome_nota(pc, acidentes)
    alteracao = {"#": 1, "b": -1}.get(escrita[1:], 0)
    linhas = ["      <harmony>", "        <root>",
              f"          <root-step>{escrita[0]}</root-step>"]
    if alteracao:
        linhas.append(f"          <root-alter>{alteracao}</root-alter>")
    linhas += ["        </root>",
               f"        <kind>{TIPOS_MUSICXML[sufixo]}</kind>",
               "      </harmony>"]
    return "\n".join(linhas)


def xml_nota(altura: int | None, unidades: int, tipo: str, pontos: int,
             acidentes: int, liga_antes: bool, liga_depois: bool) -> str:
    linhas = ["      <note>"]
    if altura is None:
        linhas.append("        <rest/>")
    else:
        escrita = nome_nota(altura % 12, acidentes)
        alteracao = {"#": 1, "b": -1}.get(escrita[1:], 0)
        linhas += ["        <pitch>", f"          <step>{escrita[0]}</step>"]
        if alteracao:
            linhas.append(f"          <alter>{alteracao}</alter>")
        linhas += [f"          <octave>{altura // 12 - 1}</octave>", "        </pitch>"]

    linhas.append(f"        <duration>{unidades}</duration>")
    if altura is not None:
        if liga_antes:
            linhas.append('        <tie type="stop"/>')
        if liga_depois:
            linhas.append('        <tie type="start"/>')
    linhas += ["        <voice>1</voice>", f"        <type>{tipo}</type>"]
    linhas += ["        <dot/>"] * pontos

    if altura is not None and (liga_antes or liga_depois):
        linhas.append("        <notations>")
        if liga_antes:
            linhas.append('          <tied type="stop"/>')
        if liga_depois:
            linhas.append('          <tied type="start"/>')
        linhas.append("        </notations>")

    linhas.append("      </note>")
    return "\n".join(linhas)


def gerar_musicxml(a: Analise, grupos: list[Grupo], compassos: list[tuple[int, int]],
                   subdiv: int, divisoes: int, batidas_compasso: int, denominador: int) -> str:
    relativa = a.tonica if a.modo == "maior" else (a.tonica + 3) % 12
    acidentes = ARMADURAS[relativa]
    modo = "major" if a.modo == "maior" else "minor"
    figura_batida = "quarter" if denominador == 4 else "eighth"

    # Ligadura entre grupos vizinhos de mesma altura (a nota foi cortada, nao repetida).
    liga_depois = [
        g.altura is not None and p.altura == g.altura and p.inicio + p.duracao == g.inicio
        for g, p in zip(grupos[1:], grupos)
    ] + [False]
    liga_antes = [False] + liga_depois[:-1]

    partes = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"'
        ' "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="4.0">',
        f"  <work><work-title>{escape(Path(a.nome).stem)}</work-title></work>",
        "  <identification><encoding><software>music-support</software></encoding>"
        "</identification>",
        '  <part-list><score-part id="P1"><part-name>Melodia</part-name></score-part>'
        "</part-list>",
        '  <part id="P1">',
    ]

    anacruse = a.fase > 0
    acorde_atual = None

    for indice, (batida, n_batidas) in enumerate(compassos):
        numero = indice if anacruse else indice + 1
        # Anacruse e ultimo compasso incompleto nao fecham a formula: marcados
        # como implicitos para o editor nao acusar compasso irregular.
        implicito = ' implicit="yes"' if n_batidas < a.batidas_por_compasso else ""
        partes.append(f'    <measure number="{numero}"{implicito}>')

        if indice == 0:
            partes += [
                "      <attributes>",
                f"        <divisions>{divisoes}</divisions>",
                f"        <key><fifths>{acidentes}</fifths><mode>{modo}</mode></key>",
                f"        <time><beats>{batidas_compasso}</beats>"
                f"<beat-type>{denominador}</beat-type></time>",
                "        <clef><sign>G</sign><line>2</line></clef>",
                "      </attributes>",
                '      <direction placement="above">',
                "        <direction-type><metronome>"
                f"<beat-unit>{figura_batida}</beat-unit>"
                f"<per-minute>{a.bpm:.0f}</per-minute></metronome></direction-type>",
                f'        <sound tempo="{a.bpm:.1f}"/>',
                "      </direction>",
            ]

        for i, g in enumerate(grupos):
            if g.compasso != indice:
                continue
            acorde = a.acordes[g.inicio // subdiv]
            if acorde != acorde_atual:
                partes.append(xml_harmonia(acorde, acidentes))
                acorde_atual = acorde

            figuras = decompor(g.duracao, divisoes)
            for j, (unidades, tipo, pontos) in enumerate(figuras):
                partes.append(xml_nota(
                    g.altura, unidades, tipo, pontos, acidentes,
                    liga_antes=liga_antes[i] or j > 0,
                    liga_depois=liga_depois[i] or j < len(figuras) - 1,
                ))

        partes.append("    </measure>")

    partes += ["  </part>", "</score-partwise>", ""]
    return "\n".join(partes)


# ---------------------------------------------------------------------- midi

def _vlq(valor: int) -> bytes:
    saida = [valor & 0x7F]
    valor >>= 7
    while valor:
        saida.append((valor & 0x7F) | 0x80)
        valor >>= 7
    return bytes(reversed(saida))


def _trilha(eventos: list[tuple[int, int, bytes]]) -> bytes:
    """eventos: (tick, prioridade, mensagem). Prioridade menor sai primeiro no mesmo tick."""
    dados = bytearray()
    anterior = 0
    for tick, _, mensagem in sorted(eventos, key=lambda e: (e[0], e[1])):
        dados += _vlq(tick - anterior) + mensagem
        anterior = tick
    dados += b"\x00\xff\x2f\x00"
    return b"MTrk" + len(dados).to_bytes(4, "big") + bytes(dados)


def gerar_midi(a: Analise, grupos: list[Grupo], subdiv: int, divisoes: int,
               batidas_compasso: int, denominador: int) -> bytes:
    def tick(slots: int) -> int:
        return round(slots * TICKS_POR_SEMINIMA / divisoes)

    microsegundos = int(round(60_000_000 / a.bpm))
    cabecalho = [
        (0, 0, b"\xff\x51\x03" + microsegundos.to_bytes(3, "big")),
        (0, 0, b"\xff\x58\x04" + bytes([batidas_compasso,
                                        denominador.bit_length() - 1, 24, 8])),
    ]

    melodia: list[tuple[int, int, bytes]] = [(0, 0, bytes([0xC0, PROGRAMA_MELODIA]))]
    # Grupos vizinhos de mesma altura sao a mesma nota cortada: junta antes de tocar.
    for g in grupos:
        if g.altura is None:
            continue
        if melodia and melodia[-1][2][:1] == b"\x80" and melodia[-1][0] == tick(g.inicio) \
                and melodia[-1][2][1] == g.altura:
            melodia.pop()  # cancela o note-off e deixa a nota seguir
        else:
            melodia.append((tick(g.inicio), 1, bytes([0x90, g.altura, 80])))
        melodia.append((tick(g.inicio + g.duracao), 0, bytes([0x80, g.altura, 0])))

    acompanhamento: list[tuple[int, int, bytes]] = [(0, 0, bytes([0xC1, PROGRAMA_ACORDES]))]
    inicio = 0
    for i in range(1, len(a.acordes) + 1):
        if i < len(a.acordes) and a.acordes[i] == a.acordes[inicio]:
            continue
        pc, sufixo = separar_acorde(a.acordes[inicio])
        t0, t1 = tick(inicio * subdiv), tick(i * subdiv)
        for intervalo in INTERVALOS[sufixo]:
            nota = 48 + pc + intervalo
            acompanhamento.append((t0, 1, bytes([0x91, nota, 55])))
            acompanhamento.append((t1, 0, bytes([0x81, nota, 0])))
        inicio = i

    trilhas = [_trilha(cabecalho), _trilha(melodia), _trilha(acompanhamento)]
    mthd = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") \
        + len(trilhas).to_bytes(2, "big") + TICKS_POR_SEMINIMA.to_bytes(2, "big")
    return mthd + b"".join(trilhas)


# ---------------------------------------------------------------------- fluxo

def gerar(caminho: Path, subdiv: int) -> tuple[Path, Path]:
    a = analisar_arquivo(caminho)

    print(f"[{caminho.name}] transcrevendo a melodia (pode demorar)...")
    slots = extrair_melodia(a, subdiv)

    batidas_compasso, denominador = (
        (6, 8) if a.compasso == "6/8" else (a.batidas_por_compasso, 4)
    )
    divisoes = subdiv * denominador // 4

    compassos = limites_compassos(len(a.acordes), a.batidas_por_compasso, a.fase)
    grupos = agrupar(slots, compassos, a.acordes, subdiv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xml = OUTPUT_DIR / f"{caminho.stem}.musicxml"
    mid = OUTPUT_DIR / f"{caminho.stem}.mid"
    xml.write_text(
        gerar_musicxml(a, grupos, compassos, subdiv, divisoes,
                       batidas_compasso, denominador),
        encoding="utf-8",
    )
    mid.write_bytes(gerar_midi(a, grupos, subdiv, divisoes, batidas_compasso, denominador))

    notas = sum(1 for g in grupos if g.altura is not None)
    print(f"[{caminho.name}] {a.tonalidade}, {a.bpm:.1f} BPM, {a.compasso}, "
          f"{len(compassos)} compassos, {notas} notas")
    return xml, mid


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Gera lead sheet (MusicXML + MIDI).")
    parser.add_argument("arquivos", nargs="*", help="arquivos a processar (padrao: input/)")
    parser.add_argument("--subdiv", type=int, default=2, choices=(1, 2, 4),
                        help="slots por batida: 1=seminima, 2=colcheia (padrao), "
                             "4=semicolcheia")
    args = parser.parse_args(argv)

    if args.arquivos:
        arquivos = [Path(a).resolve() for a in args.arquivos]
    else:
        if not INPUT_DIR.exists():
            print(f"Pasta nao encontrada: {INPUT_DIR}")
            return 1
        arquivos = sorted(p for p in INPUT_DIR.iterdir()
                          if p.is_file() and p.suffix.lower() in EXTENSOES)

    if not arquivos:
        print(f"Nenhum arquivo de audio/video em {INPUT_DIR}")
        return 1

    falhas = 0
    for arquivo in arquivos:
        try:
            xml, mid = gerar(arquivo, args.subdiv)
        except Exception as erro:  # noqa: BLE001 - um arquivo ruim nao para o lote
            falhas += 1
            print(f"[{arquivo.name}] ERRO: {erro}")
            continue
        print(f"[{arquivo.name}] gerado: {xml}")
        print(f"[{arquivo.name}] gerado: {mid}")

    return 1 if falhas == len(arquivos) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
