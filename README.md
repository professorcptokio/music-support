# music-support

Análise harmônica automática de áudio e vídeo. Dado um arquivo em `input/`, produz em `output/`:

- `analyze.py` → um **TXT** com a **tonalidade**, o **tempo**, o **compasso** e os **acordes de cada
  compasso**, separados por `|` — e, com `--letra`, a **letra transcrita** do canto;
- `partitura.py` → uma **lead sheet** em **MusicXML** (melodia transcrita + cifras) e o **MIDI**
  correspondente, para conferir de ouvido.

## Instalação

Requer Python 3.10+ e [ffmpeg](https://ffmpeg.org/) no PATH (usado para decodificar mp4, mkv e
outros contêineres de vídeo).

```
pip install -r requirements.txt
```

A transcrição de letra é opcional e só entra com `--letra`:

```
pip install faster-whisper
```

O modelo baixa sozinho no primeiro uso e fica em cache (`~/.cache/huggingface`). O padrão,
`large-v3`, ocupa cerca de 3 GB — é o que dá resultado utilizável em canto. Modelos menores
(`--modelo small`, por exemplo) baixam bem menos e rodam mais rápido, mas embaralham palavras
quando a voz está cantada. Tudo roda na CPU.

## Uso

```
python analyze.py                 # grade de acordes em TXT
python analyze.py input/x.mp4     # apenas os arquivos indicados
python analyze.py --letra         # inclui a letra transcrita
python analyze.py --letra --idioma pt              # fixa o idioma do lote
python analyze.py --letra --idioma en --idioma kyrie=la   # exceção por arquivo

python partitura.py               # lead sheet em MusicXML + MIDI
python partitura.py --subdiv 4    # resolução de semicolcheia (padrão: colcheia)
```

Sem argumentos, os dois processam tudo que houver em `input/`. Um arquivo que falhe não interrompe o
lote: o erro é impresso e os demais seguem normalmente.

Formatos aceitos: `mp3`, `mp4`, `m4a`, `wav`, `flac`, `ogg`, `aac`, `wma`, `mkv`, `mov`.

## Saída do analyze.py

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

Com `--letra`, o relatório ganha o idioma detectado no cabeçalho e uma seção com a transcrição, cada
trecho marcado com o compasso e o tempo em que entra:

```
LETRA
------------------------------------------------------------
[c.  3  0:04] ...
[c. 11  0:19] ...
------------------------------------------------------------
```

A numeração dos compassos é a mesma em toda a ferramenta (`limites_compassos`, em `analyze.py`),
então o compasso 11 da letra é o mesmo compasso 11 da grade de acordes e da partitura.

## Saída do partitura.py

Dois arquivos por música:

- `output/<nome>.musicxml` — melodia em clave de sol, com armadura, fórmula de compasso, indicação
  metronômica e as cifras acima da pauta. Abre no MuseScore, Sibelius, Finale ou flat.io; o PDF sai
  de lá, com gravação profissional.
- `output/<nome>.mid` — a mesma coisa em MIDI, melodia e acordes em canais separados. Serve para
  ouvir se a análise bate com a gravação original antes de confiar na partitura.

A anacruse e o último compasso incompleto saem marcados como `implicit`, para o editor não acusar
compasso irregular.

## Como funciona

O sinal é separado em partes harmônica e percussiva (HPSS) antes da análise, para que a harmonia
não seja contaminada pela bateria e vice-versa.

| Informação | Método |
| --- | --- |
| Tonalidade | Croma CQT do sinal harmônico, correlacionado com os perfis Krumhansl-Schmuckler de maior e menor nas 12 tônicas. |
| Tempo | `librosa.beat.beat_track` sobre o envelope de onsets do sinal percussivo. |
| Compasso | Testa 2/4, 3/4, 4/4 e 6/8 em todas as fases, pontuando o quanto o tempo forte se destaca. |
| Acordes | Croma sincronizado por batida, comparado com modelos de acorde; Viterbi para estabilizar a sequência. |
| Melodia | Saliência harmônica sobre a CQT, seguindo a voz superior; quantizada na grade de colcheias. |
| Letra | `faster-whisper` (modelo `large-v3` por padrão), sem filtro de VAD — ver abaixo. |

O `partitura.py` importa a análise do `analyze.py` (`analisar_arquivo`) e só acrescenta a melodia e a
diagramação — as duas saídas nunca divergem em tonalidade, tempo ou acordes.

Quatro pontos merecem detalhe:

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

**A melodia é a voz superior, não a mais forte.** Um detector de f0 comum (pyin, YIN) trava no baixo
quando o áudio tem acompanhamento, porque o baixo é a componente mais periódica do mix — no `kyrie`
a primeira versão devolvia exatamente as fundamentais dos acordes, na oitava 2. A busca por
saliência harmônica pega, de cada quadro, o bin mais agudo que ainda esteja bem sustentado
(`LIMIAR_TOPO`), o que sobe a linha para o registro melódico: de 90,7% para 96,6% de notas dentro da
escala e mediana em Ré4 em vez de Sol2. Subir o `fmin` do pyin não resolve — ele só pula para o
parcial seguinte.

**O filtro de VAD fica desligado, de propósito.** A detecção de atividade de voz do Whisper é
treinada em fala; em canto ela descarta o áudio inteiro. Medindo no `kyrie`, com VAD: 0 trechos
reconhecidos, e o idioma detectado vira ruído (`nn`, nynorsk). Sem VAD: 5 trechos. O preço de
desligar é que o Whisper às vezes inventa frases nos trechos instrumentais — é o lado errado menos
ruim dos dois.

**A detecção automática de idioma não é confiável em canto — informe o idioma.** Medindo os três
arquivos de teste contra o gabarito:

| arquivo | idioma real | detectado (30 s) | detectado (4 min) |
| --- | --- | --- | --- |
| holly.mp4 | inglês | en 76% ✅ | en 77% ✅ |
| LambOfGod.mp4 | inglês | en 79% ✅ | en 80% ✅ |
| kyrie.mp4 | latim | nl 18% ❌ | en 65% ❌ |

Ampliar a janela de análise (`language_detection_segments`) só melhora a *confiança*, não a
*resposta*: o erro passou de "holandês com 18%", que é ruído de introdução instrumental, para
"inglês com 65%", que é voz de verdade classificada errado. Restringir os candidatos a um conjunto
pequeno também não resolve, porque o inglês vence o latim dentro do conjunto.

O motivo aparece na distribuição completa: o modelo dá 2,14% de latim para o `kyrie`, que é em
latim, e 2,89% para o `holly`, que é em inglês. A probabilidade de latim é **maior na música
errada** — o sinal não existe, e nenhum pós-processamento em cima dele funciona. Por isso o código
não tenta corrigir a detecção: ele avisa sempre que o idioma não foi informado (a confiança não
separa acerto de erro, então não há limiar que preste) e deixa você dizer qual é.

Como um lote costuma ser misto, o `--idioma` aceita exceção por arquivo:

```
python analyze.py --letra --idioma en --idioma kyrie=la
```

O valor solto (`en`) vale para todo o lote; o par (`kyrie=la`) sobrepõe só naquele arquivo, casando
pelo nome sem extensão.

Os parâmetros desses vieses estão no topo de `analyze.py` (`BONUS_DIATONICO`, `BONUS_PERMANENCIA`,
`QUALIDADES`) e de `partitura.py` (`LIMIAR_TOPO`, `LIMIAR_VOZ`, `NOTA_MIN`/`NOTA_MAX`), caso queira
calibrar para um repertório específico.

## Limitações

O resultado é uma estimativa, e o próprio TXT diz isso no rodapé. Vale conferir de ouvido antes de
usar em ensaio ou partitura.

Uma medição contra gabarito corrigido à mão (`holly.mp4`, 54 compassos, único arquivo de teste com
grade compasso a compasso conferida) dá a ordem de grandeza: **tonalidade, tempo e compasso
acertaram**, mas nos acordes só 48% dos compassos trazem a cifra certa em primeiro lugar, e 63%
a trazem em algum lugar da célula. Ou seja, a grade harmônica serve como ponto de partida, não como
cifra pronta — a diferença de confiabilidade entre o cabeçalho e os acordes é grande.

Na prática:

- Inversões e baixo não são detectados: um `G/B` sai como `G`.
- Modulações não são acompanhadas — a tonalidade é única para a música inteira, e os acordes depois
  de uma modulação tendem a piorar, porque o bônus diatônico continua apontando para a tonalidade
  original.
- Andamento variável (rubato, acelerando) degrada o rastreio de batidas e, por consequência, o
  alinhamento dos compassos.
- Mudanças de fórmula de compasso no meio da música não são detectadas.
- Trechos com voz solo, muita reverberação ou percussão densa produzem acordes menos confiáveis.

Na melodia, especificamente:

- É a parte menos confiável de tudo. Não há separação de fontes: quando o arranjo é denso, ou quando
  um instrumento agudo passa por cima do canto, a linha extraída pula para ele.
- O ritmo é quantizado na grade de colcheias (ou semicolcheias com `--subdiv 4`). Síncopes finas,
  quiálteras e notas fora da grade são arredondadas.
- Ornamentos, glissandos e vibrato viram notas isoladas espúrias.
- Trate o MusicXML como rascunho para editar no MuseScore, não como partitura final.

Na letra:

- O Whisper foi treinado em fala, não em canto. Melismas, vibrato, notas longas e coro em várias
  vozes derrubam bastante a precisão — espere corrigir palavras à mão.
- Em trechos só instrumentais ele tende a inventar frases. O filtro de VAD corta a maior parte
  disso, mas não tudo.
- A detecção automática de idioma erra muito em canto, e erra com confiança alta. Passe `--idioma`
  sempre que souber; para latim é praticamente obrigatório (ver a medição acima).
- Os tempos vêm do Whisper, não do alinhamento com as batidas: o compasso indicado marca onde o
  trecho começa, com precisão de mais ou menos um compasso. Não é sincronia de karaokê.

Um bom sinal de erro é a repetição: quando a música tem um ciclo regular, os compassos que destoam
do padrão nas outras voltas costumam ser justamente os que a análise errou.
