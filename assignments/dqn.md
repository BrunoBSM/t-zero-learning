# Atividade: Deep Q-Networks (DQN)

Individual ou em dupla (identifique a dupla no relatório).

Você recebe uma implementação quase completa de DQN ([algorithms/dqn.py](../algorithms/dqn.py)) e integrada ao harness de
treinamento deste repositório (https://github.com/BrunoBSM/t-zero-learning). O algoritmo inteiro — replay buffer, alvo de TD, rollout
epsilon-greedy, target network — vive nesse único arquivo; o framework fornece a
infraestrutura de execução (configs, seeds, logging, diretórios de runs). Seu trabalho
tem duas partes: uma parte curta de implementação e um relatório experimental —
**o relatório é onde está a maior parte da nota.**

## Preparação

A partir da raiz do repositório:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

As métricas vão para o [Weights & Biases](https://wandb.ai). Crie um arquivo `.env` na raiz do repositório:

```
WANDB_PROJECT=dqn-assignment
```

e rode `wandb login` uma vez.

## Partes 1 e 2 — Implementação

Complete os três blocos marcados com `YOUR CODE HERE` em `algorithms/dqn.py`:

- **Parte 1a** — `ReplayBuffer.add`: armazena uma transição, sobrescrevendo a mais
  antiga (FIFO) quando o buffer está cheio.
- **Parte 1b** — `ReplayBuffer.sample`: minibatch uniforme aleatório das transições
  armazenadas.
- **Parte 2** — `compute_td_targets`: o alvo de TD de um passo — pense em tudo de
  que o alvo precisa e de onde vem cada peça.

Cada bloco tem poucas linhas. Verifique com:

```bash
python -m pytest tests/test_dqn.py
```

Todos os testes devem passar antes de você começar a Parte 3. Depois treine (sempre a
partir da raiz do repositório):

```bash
python train.py --config dqn_cartpole
```

Os hiperparâmetros estão em [configs/dqn_cartpole.yml](../configs/dqn_cartpole.yml);
qualquer um deles pode ser alterado por run com `--override` (veja abaixo) — você não
deve precisar editar o arquivo de config.

Um treino completo no CartPole leva cerca de 10–20 minutos em CPU de notebook e deve
atingir retorno episódico próximo de 500 (o máximo). Se não atingir, algo está errado —
os testes passarem é condição necessária, mas não suficiente.

## Os gráficos com que você vai trabalhar

Todo run registra estes gráficos no wandb; seu relatório deve se basear neles
(screenshots ou links de relatório do wandb, sempre com os runs identificados):

| Gráfico | O que ele diz |
|---|---|
| `charts/episodic_return_mean_last100` | a curva de aprendizado — retorno médio dos últimos 100 episódios |
| `losses/td_loss` | quão longe Q(s,a) está do alvo de TD nos batches amostrados |
| `losses/q_values` | Q médio predito — observe divergência ou crescimento descontrolado |
| `charts/epsilon` | o cronograma de exploração efetivamente usado |
| `charts/episodic_length_mean_last100` | duração dos episódios (no CartPole, ≡ retorno) |
| `eval/mean_return`, `eval/std_return` | avaliação greedy final de 10 episódios |

Um hábito central que este trabalho treina: **nunca leia um gráfico isoladamente.** A
curva de retorno diz *se* algo deu errado; `td_loss` e `q_values` juntos costumam dizer
*o quê*.

## Parte 3 — Relatório experimental

Para cada questão abaixo, siga este protocolo:

1. **Preveja** (antes de rodar — no máximo 2–3 frases): o que você espera que os
   gráficos indicados façam ao longo da sua varredura, e por quê?
2. **Varredura (sweep)**: escolha **pelo menos dois valores** do hiperparâmetro (à sua
   escolha, cobrindo de pequeno → grande o bastante para expor o comportamento), além
   do baseline. Rode cada configuração; quando viável, rode 2 seeds nas configurações
   das quais seu argumento depende (`--override seed=2`).
3. **Reporte os gráficos**: para cada questão, inclua os gráficos indicados, com todos
   os valores da varredura sobrepostos e os runs identificados.
4. **Explique**: 1–2 parágrafos sobre o *mecanismo* por trás do que você observou —
   "piorou" é uma descrição, não uma explicação. As explicações devem referenciar os
   gráficos ("o gráfico de q_values mostra …, o que significa …") e dizer se a sua
   previsão se confirmou.

Os overrides funcionam assim (qualquer chave de
[configs/dqn_cartpole.yml](../configs/dqn_cartpole.yml)):

```bash
python train.py --config dqn_cartpole --override dqn.target_network_frequency=1 seed=2
```

**Q1 — Frequência de sincronização da target network**
(`dqn.target_network_frequency`; 1 = um alvo novo a cada passo, grande = um alvo quase
congelado).

Gráficos a reportar: `charts/episodic_return_mean_last100`, `losses/td_loss`,
`losses/q_values`.

Qual o papel da target network? Por que a curva de retorno pode degradar enquanto o
`td_loss` continua parecendo "bem"? O que `losses/q_values` faz nos seus valores
extremos, e por quê?

**Q2 — Tamanho do replay buffer** (`dqn.buffer_size`; experimente de minúsculo a
generoso).

Gráficos a reportar: `charts/episodic_return_mean_last100`, `losses/q_values`.

Quais dois problemas distintos um buffer muito pequeno causa? (Dica: pense tanto na
*correlação* das amostras dentro de um minibatch quanto em *quais dados a rede
consegue rever*.)

**Q3 — Sua escolha. (EXTRA)** Escolha qualquer outro hiperparâmetro (`dqn.gamma`,
`dqn.learning_rate`, `dqn.exploration_fraction`, `dqn.end_e`, `dqn.batch_size`,
`dqn.train_frequency`, ...), faça a varredura da mesma forma e diga *quais gráficos
você escolheu reportar e por quê* — a escolha dos gráficos faz parte da resposta aqui
(p. ex., uma varredura de exploração sem `charts/epsilon` está incompleta). Não mude apenas a seed.

## Entregáveis

Um relatório em
PDF (máx. 3 páginas) com as três seções tendo previsão → varredura → gráficos → explicação, contendo um link para o seu fork com o `algorithms/dqn.py` completo
