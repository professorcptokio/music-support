"""Analisa arquivos de audio/video da pasta input e gera um TXT por arquivo na pasta output.

Cada TXT contem:
  - TONALIDADE da cancao (Krumhansl-Schmuckler sobre o croma)
  - TEMPO em BPM
  - COMPASSO estimado (2/4, 3/4, 4/4, 6/8)
  - Os possiveis acordes de cada compasso, separados por |

Com --letra, transcreve tambem o canto (faster-whisper) e ancora cada trecho no
compasso em que ele entra.

Uso:
    python analyze.py                 # processa tudo que houver em input/
    python analyze.py input/x.mp4     # processa apenas os arquivos indicados
    python analyze.py --letra         # inclui a letra transcrita
    python analyze.py --letra --idioma pt
"""

from __future__ import annotations

import argparse
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
class Fala:
    """Um trecho de letra transcrito, ancorado no tempo e no compasso."""

    inicio: float     # em segundos
    fim: float
    texto: str
    compasso: int     # numeracao de limites_compassos, igual a do resto do projeto


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


def limites_compassos(n_batidas: int, bpc: int, fase: int) -> list[tuple[int, int]]:
    """Lista de (batida inicial, quantidade de batidas). O primeiro pode ser anacruse.

    E a numeracao unica do projeto: relatorio TXT, letra e partitura usam esta mesma
    divisao, entao o compasso 12 e o mesmo compasso nos tres.
    """
    limites = []
    if fase:
        limites.append((0, fase))
    inicio = fase
    while inicio < n_batidas:
        limites.append((inicio, min(bpc, n_batidas - inicio)))
        inicio += bpc
    return limites


def agrupar_por_compasso(acordes: list[str], batidas_por_compasso: int, fase: int) -> list[list[str]]:
    """Quebra a sequencia de batidas em compassos, listando ate 2 acordes por compasso."""
    compassos = [acordes[i:i + n]
                 for i, n in limites_compassos(len(acordes), batidas_por_compasso, fase)]

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


# -------------------------------------------------------------------------- letra

def carregar_modelo(nome: str):
    """Carrega o Whisper uma vez so, para o lote inteiro reaproveitar."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as erro:
        raise RuntimeError(
            "para transcrever a letra instale o motor de reconhecimento:\n"
            "    pip install faster-whisper"
        ) from erro

    print(f"carregando o modelo '{nome}' (o primeiro uso baixa os pesos)...")
    return WhisperModel(nome, device="cpu", compute_type="int8")


def transcrever(a: Analise, modelo, idioma: str | None) -> tuple[list[Fala], str, float]:
    """Transcreve o canto e ancora cada trecho no compasso em que ele comeca."""
    audio = librosa.resample(a.y, orig_sr=a.sr, target_sr=16000)
    audio = audio / max(1.0, float(np.abs(audio).max()))

    segmentos, info = modelo.transcribe(
        audio,
        language=idioma,
        beam_size=5,
        # O VAD e treinado em fala e descarta canto: no kyrie derrubou a faixa
        # inteira (0 trechos com VAD contra 5 sem). Por isso fica desligado, ao
        # custo de o Whisper as vezes inventar frases em trecho instrumental.
        vad_filter=False,
        condition_on_previous_text=False,  # sem isso o Whisper entra em loop no refrao
    )

    inicios = [a.tempos_batidas[batida]
               for batida, _ in limites_compassos(len(a.acordes),
                                                  a.batidas_por_compasso, a.fase)]

    falas = []
    for s in segmentos:
        texto = s.text.strip()
        if texto:
            compasso = int(np.searchsorted(inicios, s.start, side="right"))
            falas.append(Fala(s.start, s.end, texto, max(1, compasso)))
    return falas, info.language, info.language_probability


# ------------------------------------------------------------------------- saida

def formatar_relatorio(a: Analise, compassos: list[list[str]],
                       falas: list[Fala] | None = None, rotulo: str | None = None) -> str:
    linhas = [
        f"ARQUIVO....: {a.nome}",
        f"DURACAO....: {int(a.duracao // 60)}:{int(a.duracao % 60):02d}",
        f"TONALIDADE.: {a.tonalidade}",
        f"TEMPO......: {a.bpm:.1f} BPM",
        f"COMPASSO...: {a.compasso}",
    ]
    if rotulo:
        linhas.append(f"IDIOMA.....: {rotulo}")
    linhas += ["", "ACORDES POR COMPASSO", "-" * 60]

    por_linha = 4
    for i in range(0, len(compassos), por_linha):
        bloco = compassos[i:i + por_linha]
        celulas = " | ".join(" ".join(c) for c in bloco)
        linhas.append(f"{i + 1:>4}: | {celulas} |")

    linhas += ["-" * 60, f"Total de compassos: {len(compassos)}"]

    if falas is not None:
        linhas += ["", "LETRA", "-" * 60]
        if falas:
            for f in falas:
                marca = f"c.{f.compasso:>3}  {int(f.inicio // 60)}:{int(f.inicio % 60):02d}"
                linhas.append(f"[{marca}] {f.texto}")
        else:
            linhas.append("(nenhum canto reconhecido - faixa instrumental?)")
        linhas.append("-" * 60)

    linhas += ["", "Analise automatica (estimativa) - confira de ouvido antes de usar."]
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


def analisar(caminho: Path, modelo=None, idioma: str | None = None) -> str:
    a = analisar_arquivo(caminho)
    compassos = agrupar_por_compasso(a.acordes, a.batidas_por_compasso, a.fase)

    falas = rotulo = None
    if modelo is not None:
        print(f"[{caminho.name}] transcrevendo a letra (pode demorar)...")
        falas, detectado, confianca = transcrever(a, modelo, idioma)
        if idioma:
            rotulo = f"{detectado} (informado)"
        else:
            # A confianca nao separa acerto de erro em canto: no kyrie o modelo
            # deu ingles com 65% para uma faixa em latim. Por isso o aviso sai
            # sempre que o idioma nao foi informado, e nao abaixo de um limiar.
            rotulo = f"{detectado} (detectado, confianca {confianca:.0%})"
            print(f"[{caminho.name}] AVISO: idioma detectado automaticamente "
                  f"({detectado}, {confianca:.0%}). Em canto isso erra com "
                  f"frequencia - use --idioma se souber qual e.")

    return formatar_relatorio(a, compassos, falas, rotulo)


def resolver_idiomas(valores: list[str]) -> tuple[str | None, dict[str, str]]:
    """Le os --idioma: 'en' vira padrao do lote, 'kyrie=la' vira excecao do arquivo.

    Um lote costuma ser misto (duas musicas em ingles e uma em latim, por exemplo)
    e a deteccao automatica nao resolve isso, entao da para dizer arquivo a arquivo.
    """
    padrao, por_arquivo = None, {}
    for valor in valores:
        if "=" in valor:
            nome, _, codigo = valor.partition("=")
            if not nome.strip() or not codigo.strip():
                raise ValueError(f"--idioma invalido: {valor!r} (use ARQUIVO=CODIGO)")
            por_arquivo[Path(nome.strip()).stem.lower()] = codigo.strip()
        elif padrao is not None:
            raise ValueError(f"--idioma geral informado duas vezes: {padrao!r} e {valor!r}")
        else:
            padrao = valor.strip()
    return padrao, por_arquivo


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Gera um TXT com tonalidade, tempo, compasso e acordes de cada musica.")
    parser.add_argument("arquivos", nargs="*", help="arquivos a processar (padrao: input/)")
    parser.add_argument("--letra", action="store_true",
                        help="transcreve o canto e inclui a letra no relatorio")
    parser.add_argument("--modelo", default="large-v3",
                        help="modelo Whisper usado com --letra (padrao: large-v3)")
    parser.add_argument("--idioma", action="append", default=[], metavar="CODIGO|ARQUIVO=CODIGO",
                        help="idioma cantado: 'en' vale para todos; 'kyrie=la' so para esse "
                             "arquivo. Pode repetir. Padrao: detectar (erra muito em canto)")
    args = parser.parse_args(argv)

    try:
        idioma_padrao, idioma_por_arquivo = resolver_idiomas(args.idioma)
    except ValueError as erro:
        parser.error(str(erro))

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

    modelo = carregar_modelo(args.modelo) if args.letra else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    falhas = 0
    for arquivo in arquivos:
        try:
            idioma = idioma_por_arquivo.get(arquivo.stem.lower(), idioma_padrao)
            relatorio = analisar(arquivo, modelo, idioma)
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
