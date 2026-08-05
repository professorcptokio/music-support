# music-support

Análise harmônica automática de áudio e vídeo. Dado um arquivo em `input/`, gera em `output/` um TXT
com o mesmo nome contendo a **tonalidade**, o **tempo**, o **compasso** e os **acordes de cada
compasso**, separados por `|`.

## Instalação

Requer Python 3.10+ e [ffmpeg](https://ffmpeg.org/) no PATH (usado para decodificar mp4, mkv e
outros contêineres de vídeo).

```
pip install -r requirements.txt
```

## Uso

```
python analyze.py                 # processa todos os arquivos de input/
python analyze.py input/x.mp4     # processa apenas os arquivos indicados
```

Cada arquivo processado vira `output/<nome-do-arquivo>.txt`. Um arquivo que falhe não interrompe o
lote: o erro é impresso e os demais seguem normalmente.

Formatos aceitos: `mp3`, `mp4`, `m4a`, `wav`, `flac`, `ogg`, `aac`, `wma`, `mkv`, `mov`.

## Saída

```
ARQUIVO....: kyrie.mp4
DURACAO....: 1:46
TONALIDADE.: G maior
TEMPO......: 129.2 BPM
COMPASSO...: 4/4

ACORDES POR COMPASSO
------------------------------------------------------------
   1: | Em | Em | G | G |
   5: | G | C | D | Em |
   9: | Em | Em | D G | G |
...
```

O número à esquerda é o primeiro compasso da linha, que traz quatro compassos. Quando duas
harmonias ocupam claramente o mesmo compasso, as duas aparecem na célula (`| D G |`). Se a música
começa em anacruse, as batidas anteriores ao primeiro tempo forte formam o compasso 1, mais curto.

## Como funciona

O sinal é separado em partes harmônica e percussiva (HPSS) antes da análise, para que a harmonia
não seja contaminada pela bateria e vice-versa.

| Informação | Método |
| --- | --- |
| Tonalidade | Croma CQT do sinal harmônico, correlacionado com os perfis Krumhansl-Schmuckler de maior e menor nas 12 tônicas. |
| Tempo | `librosa.beat.beat_track` sobre o envelope de onsets do sinal percussivo. |
| Compasso | Testa 2/4, 3/4, 4/4 e 6/8 em todas as fases, pontuando o quanto o tempo forte se destaca. |
| Acordes | Croma sincronizado por batida, comparado com modelos de acorde; Viterbi para estabilizar a sequência. |

Dois pontos merecem detalhe:

**A ordem importa.** O compasso é deduzido *depois* dos acordes, não antes. A troca de acorde é a
pista mais confiável do início do compasso — mais que o acento rítmico — e por isso pesa metade da
pontuação de fase, contra um quarto para o acento e um quarto para a mudança de croma. Numa versão
que decidia o compasso só pelo acento, a fase saía deslocada em uma batida e quase todo compasso
era impresso como `X Y`, com a troca de harmonia caindo no fim do compasso em vez do começo.

**O vocabulário de acordes é enviesado de propósito.** Reconhece maj, min, 7, m7, maj7, sus4 e dim
nas 12 tônicas, mas tríades simples recebem peso maior que as extensões, de modo que uma sétima só
vence quando está de fato evidente no croma. Acordes do campo harmônico da tonalidade detectada
ganham um bônus pequeno, e permanecer no mesmo acorde de uma batida para a outra também — sem isso
a sequência oscila a cada batida.

Os parâmetros desses vieses estão no topo de `analyze.py` (`BONUS_DIATONICO`, `BONUS_PERMANENCIA`,
`QUALIDADES`), caso queira calibrar para um repertório específico.

## Limitações

O resultado é uma estimativa, e o próprio TXT diz isso no rodapé. Vale conferir de ouvido antes de
usar em ensaio ou partitura. Na prática:

- Inversões e baixo não são detectados: um `G/B` sai como `G`.
- Modulações não são acompanhadas — a tonalidade é única para a música inteira, e os acordes depois
  de uma modulação tendem a piorar, porque o bônus diatônico continua apontando para a tonalidade
  original.
- Andamento variável (rubato, acelerando) degrada o rastreio de batidas e, por consequência, o
  alinhamento dos compassos.
- Mudanças de fórmula de compasso no meio da música não são detectadas.
- Trechos com voz solo, muita reverberação ou percussão densa produzem acordes menos confiáveis.

Um bom sinal de erro é a repetição: quando a música tem um ciclo regular, os compassos que destoam
do padrão nas outras voltas costumam ser justamente os que a análise errou.
