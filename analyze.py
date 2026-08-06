"""Analisa arquivos de audio/video da pasta input e gera um TXT por arquivo na pasta output.

Cada TXT contem:
  - TONALIDADE da cancao (Krumhansl-Schmuckler sobre o croma)
  - TEMPO em BPM
  - COMPASSO estimado (2/4, 3/4, 4/4, 6/8)
  - Os possiveis acordes de cada compasso, separados por |

Uso:
    python analyze.py                 # processa tudo que houver em input/
    python analyze.py input/x.mp4     # processa apenas os arquivos indicados
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

EXTENSOES = {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma", ".mkv", ".mov"}

SR = 22050
NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Perfis de Krumhansl-Schmuckler para deteccao de tonalidade.
PERFIL_MAIOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PERFIL_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Vocabulario de acordes: sufixo -> (intervalos, peso).
# O peso favorece triades simples para que 7as so aparecam quando forem evidentes.
QUALIDADES = [
    ("", (0, 4, 7), 1.00),
    ("m", (0, 3, 7), 1.00),
    ("7", (0, 4, 7, 10), 0.95),
    ("m7", (0, 3, 7, 10), 0.95),
    ("maj7", (0, 4, 7, 11), 0.93),
    ("sus4", (0, 5, 7), 0.92),
    ("dim", (0, 3, 6), 0.90),
]

# Graus (semitons a partir da tonica) e qualidade esperada do campo harmonico.
DIATONICOS_MAIOR = {0: "", 2: "m", 4: "m", 5: "", 7: "", 9: "m", 11: "dim"}
DIATONICOS_MENOR = {0: "m", 2: "dim", 3: "", 5: "m", 7: "m", 8: "", 10: ""}

BONUS_DIATONICO = 0.06   # empurra o acorde para dentro da tonalidade detectada
BONUS_PERMANENCIA = 0.22  # penaliza trocas de acorde a cada batida


@dataclass
class Analise:
    """Resultado cru da analise, antes de virar TXT ou partitura."""

    nome: str
    duracao: float
    y: np.ndarray = field(repr=False)
    sr: int
    tonalidade: str
    tonica: int          # semitons a partir de C
    modo: str            # "maior" ou "menor"
    bpm: float
    tempos_batidas: np.ndarray = field(repr=False)  # em segundos
    acordes: list[str]   # um por batida; acordes[i] vale de tempos_batidas[i] a [i+1]
    batidas_por_compasso: int
    fase: int            # batidas de anacruse antes do primeiro tempo forte
    compasso: str        # "4/4", "3/4", ...


# --------------------------------------------------------------------------- audio

def carregar_audio(caminho: Path) -> tuple[np.ndarray, int]:
    """Decodifica o arquivo em mono. Usa ffmpeg quando disponivel (cobre mp4/mkv)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-i", str(caminho),
                 "-vn", "-ac", "1", "-ar", str(SR), str(wav)],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and wav.exists():
                return librosa.load(wav, sr=SR, mono=True)
            print(f"  ffmpeg falhou ({proc.stderr.strip()[:120]}), tentando via librosa...")
    return librosa.load(caminho, sr=SR, mono=True)


# ------------------------------------------------------------------------ analise

def detectar_tonalidade(croma: np.ndarray) -> tuple[str, int, str]:
    """Retorna (nome da tonalidade, tonica em semitons, modo)."""
    perfil = croma.mean(axis=1)
    perfil = perfil - perfil.mean()

    melhor = (-np.inf, 0, "maior")
    for tonica in range(12):
        for modo, referencia in (("maior", PERFIL_MAIOR), ("menor", PERFIL_MENOR)):
            ref = np.roll(referencia, tonica)
            ref = ref - ref.mean()
            denom = np.linalg.norm(perfil) * np.linalg.norm(ref)
            score = float(perfil @ ref / denom) if denom else -np.inf
            if score > melhor[0]:
                melhor = (score, tonica, modo)

    _, tonica, modo = melhor
    sufixo = "maior" if modo == "maior" else "menor"
    return f"{NOTAS[tonica]} {sufixo}", tonica, modo


def detectar_compasso(forca_batidas: np.ndarray, novidade: np.ndarray,
                      mudancas: np.ndarray) -> tuple[int, int]:
    """Escolhe (batidas por compasso, batida inicial) pela saliencia dos tempos fortes.

    O tempo forte concentra tres pistas: acento ritmico, mudanca de croma e,
    principalmente, troca de acorde - por isso `mudancas` pesa mais.
    """
    if len(forca_batidas) < 8:
        return 4, 0

    f = forca_batidas / (forca_batidas.max() or 1.0)
    n = novidade / (novidade.max() or 1.0)
    sinal = 0.25 * f + 0.25 * n + 0.5 * mudancas

    # Leve preferencia por 4/4, que e de longe o mais comum.
    preferencia = {2: 0.97, 3: 1.00, 4: 1.03, 6: 0.96}

    melhor = (-np.inf, 4, 0)
    for m in (2, 3, 4, 6):
        for fase in range(m):
            fortes = sinal[fase::m]
            mascara = np.ones(len(sinal), dtype=bool)
            mascara[fase::m] = False
            fracos = sinal[mascara]
            if len(fortes) < 2 or len(fracos) < 2:
                continue
            score = (fortes.mean() - fracos.mean()) * preferencia[m]
            if score > melhor[0]:
                melhor = (score, m, fase)

    return melhor[1], melhor[2]


def montar_modelos() -> tuple[np.ndarray, list[str], list[float], list[tuple[int, str]]]:
    modelos, nomes, pesos, ident = [], [], [], []
    for raiz in range(12):
        for sufixo, intervalos, peso in QUALIDADES:
            vetor = np.zeros(12)
            for i, intervalo in enumerate(intervalos):
                # Fundamental e quinta pesam mais que as extensoes.
                vetor[(raiz + intervalo) % 12] = 1.0 if i < 3 else 0.8
            modelos.append(vetor / np.linalg.norm(vetor))
            nomes.append(f"{NOTAS[raiz]}{sufixo}")
            pesos.append(peso)
            ident.append((raiz, sufixo))
    return np.array(modelos), nomes, pesos, ident


def reconhecer_acordes(croma_batidas: np.ndarray, tonica: int, modo: str) -> list[str]:
    """Um acorde por batida, com Viterbi para evitar trocas espurias."""
    modelos, nomes, pesos, ident = montar_modelos()

    diatonicos = DIATONICOS_MAIOR if modo == "maior" else DIATONICOS_MENOR
    bonus = np.array([
        BONUS_DIATONICO if diatonicos.get((raiz - tonica) % 12) == sufixo.rstrip("7").replace("maj", "")
        else 0.0
        for raiz, sufixo in ident
    ])

    normas = np.linalg.norm(croma_batidas, axis=0)
    normas[normas == 0] = 1.0
    emissao = (modelos @ croma_batidas) / normas          # (acordes, batidas)
    emissao = emissao * np.array(pesos)[:, None] + bonus[:, None]

    n_acordes, n_batidas = emissao.shape
    custo = emissao[:, 0].copy()
    caminho = np.zeros((n_acordes, n_batidas), dtype=int)

    for t in range(1, n_batidas):
        anterior_melhor = int(np.argmax(custo))
        base = custo[anterior_melhor]
        # Ficar no mesmo acorde ganha um bonus; mudar paga o preco da melhor origem.
        ficar = custo + BONUS_PERMANENCIA
        mudar = np.full(n_acordes, base)
        escolha_ficar = ficar > mudar
        caminho[:, t] = np.where(escolha_ficar, np.arange(n_acordes), anterior_melhor)
        custo = np.where(escolha_ficar, ficar, mudar) + emissao[:, t]

    sequencia = [0] * n_batidas
    sequencia[-1] = int(np.argmax(custo))
    for t in range(n_batidas - 1, 0, -1):
        sequencia[t - 1] = caminho[sequencia[t], t]

    return [nomes[i] for i in sequencia]


def agrupar_por_compasso(acordes: list[str], batidas_por_compasso: int, fase: int) -> list[list[str]]:
    """Quebra a sequencia de batidas em compassos, listando ate 2 acordes por compasso."""
    compassos = []
    inicio = fase
    if fase:  # anacruse
        compassos.append(acordes[:fase])
    while inicio < len(acordes):
        compassos.append(acordes[inicio:inicio + batidas_por_compasso])
        inicio += batidas_por_compasso

    resultado = []
    for bloco in compassos:
        if not bloco:
            continue
        grupos: list[list[str]] = []
        for acorde in bloco:
            if grupos and grupos[-1][0] == acorde:
                grupos[-1].append(acorde)
            else:
                grupos.append([acorde])
        minimo = max(1, len(bloco) // 3)
        relevantes = [g[0] for g in grupos if len(g) >= minimo] or [grupos[0][0]]
        resultado.append(relevantes[:2])
    return resultado


# ------------------------------------------------------------------------- saida

def formatar_relatorio(nome: str, tonalidade: str, bpm: float, compasso: str,
                       compassos: list[list[str]], duracao: float) -> str:
    linhas = [
        f"ARQUIVO....: {nome}",
        f"DURACAO....: {int(duracao // 60)}:{int(duracao % 60):02d}",
        f"TONALIDADE.: {tonalidade}",
        f"TEMPO......: {bpm:.1f} BPM",
        f"COMPASSO...: {compasso}",
        "",
        "ACORDES POR COMPASSO",
        "-" * 60,
    ]

    por_linha = 4
    for i in range(0, len(compassos), por_linha):
        bloco = compassos[i:i + por_linha]
        celulas = " | ".join(" ".join(c) for c in bloco)
        linhas.append(f"{i + 1:>4}: | {celulas} |")

    linhas += [
        "-" * 60,
        f"Total de compassos: {len(compassos)}",
        "",
        "Analise automatica (estimativa) - confira de ouvido antes de usar.",
    ]
    return "\n".join(linhas) + "\n"


def analisar_arquivo(caminho: Path) -> "Analise":
    """Roda a analise completa e devolve os dados crus, sem formatacao.

    E o ponto de entrada usado tanto pelo relatorio TXT quanto pelo partitura.py.
    """
    print(f"[{caminho.name}] carregando audio...")
    y, sr = carregar_audio(caminho)
    duracao = librosa.get_duration(y=y, sr=sr)

    print(f"[{caminho.name}] separando harmonia e detectando batidas...")
    y_harm, y_perc = librosa.effects.hpss(y)

    envelope = librosa.onset.onset_strength(y=y_perc, sr=sr)
    bpm, batidas = librosa.beat.beat_track(onset_envelope=envelope, sr=sr, trim=False)
    bpm = float(np.atleast_1d(bpm)[0])

    croma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    tonalidade, tonica, modo = detectar_tonalidade(croma)

    if len(batidas) < 4:
        raise RuntimeError("nao foi possivel detectar batidas suficientes")

    # pad=False: o segmento i vai da batida i ate a batida i+1, entao o croma
    # fica alinhado batida a batida com `forca` (a ultima batida so fecha o
    # ultimo segmento e nao gera coluna propria).
    croma_batidas = librosa.util.sync(croma, batidas, aggregate=np.median, pad=False)
    forca = envelope[np.clip(batidas[:-1], 0, len(envelope) - 1)]
    n = min(croma_batidas.shape[1], len(forca))
    croma_batidas, forca = croma_batidas[:, :n], forca[:n]

    # Novidade harmonica: quanto o croma muda de uma batida para a outra.
    diffs = np.linalg.norm(np.diff(croma_batidas, axis=1), axis=0)
    novidade = np.concatenate([[float(diffs.mean())], diffs])

    print(f"[{caminho.name}] reconhecendo acordes...")
    acordes = reconhecer_acordes(croma_batidas, tonica, modo)

    # As trocas de acorde sao a pista mais forte do tempo forte, entao o
    # compasso e a fase saem depois da harmonia.
    mudancas = np.array([0.0] + [float(a != b) for a, b in zip(acordes, acordes[1:])])

    batidas_por_compasso, fase = detectar_compasso(forca, novidade, mudancas)
    compasso = {2: "2/4", 3: "3/4", 4: "4/4", 6: "6/8"}[batidas_por_compasso]

    return Analise(
        nome=caminho.name,
        duracao=duracao,
        y=y,
        sr=sr,
        tonalidade=tonalidade,
        tonica=tonica,
        modo=modo,
        bpm=bpm,
        tempos_batidas=librosa.frames_to_time(batidas, sr=sr),
        acordes=acordes,
        batidas_por_compasso=batidas_por_compasso,
        fase=fase,
        compasso=compasso,
    )


def analisar(caminho: Path) -> str:
    a = analisar_arquivo(caminho)
    compassos = agrupar_por_compasso(a.acordes, a.batidas_por_compasso, a.fase)
    return formatar_relatorio(a.nome, a.tonalidade, a.bpm, a.compasso, compassos, a.duracao)


def main(argv: list[str]) -> int:
    if argv:
        arquivos = [Path(a).resolve() for a in argv]
    else:
        if not INPUT_DIR.exists():
            print(f"Pasta nao encontrada: {INPUT_DIR}")
            return 1
        arquivos = sorted(p for p in INPUT_DIR.iterdir()
                          if p.is_file() and p.suffix.lower() in EXTENSOES)

    if not arquivos:
        print(f"Nenhum arquivo de audio/video em {INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    falhas = 0
    for arquivo in arquivos:
        try:
            relatorio = analisar(arquivo)
        except Exception as erro:  # noqa: BLE001 - um arquivo ruim nao para o lote
            falhas += 1
            print(f"[{arquivo.name}] ERRO: {erro}")
            continue
        destino = OUTPUT_DIR / f"{arquivo.stem}.txt"
        destino.write_text(relatorio, encoding="utf-8")
        print(f"[{arquivo.name}] gerado: {destino}")

    return 1 if falhas == len(arquivos) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
