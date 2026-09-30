# Atividade: Policy Gradient com Actor-Critic (A2C)

Individual ou em dupla (identifique a dupla no relatório).

Na aula vimos o A3C. O que você vai implementar é o **A2C** — o mesmo algoritmo
(retorno de n passos, ator + crítico, bônus de entropia, vários atores em paralelo),
mas com os atores avançando em *sincronia*: `num_envs` ambientes dão `num_steps` passos
cada, e o lote inteiro produz **um** passo de gradiente. Sem threads, sem gradientes
atrasados — o resto é idêntico.

Você recebe uma implementação quase completa, integrada ao harness de treinamento deste
repositório (https://github.com/BrunoBSM/t-zero-learning). O algoritmo vive em dois
arquivos:

- [algorithms/a2c.py](../algorithms/a2c.py) — rollout, retornos, perdas, atualização;
- [networks/discrete_actor_critic.py](../networks/discrete_actor_critic.py) — a rede
  ator-crítico (política categórica + V(s)).

O framework fornece a infraestrutura de execução (configs, seeds, logging, diretórios de
runs). Seu trabalho tem duas partes: uma parte curta de implementação e um relatório
experimental.

## Preparação

Igual à atividade de DQN. A partir da raiz do repositório:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No `.env`, use um projeto novo no [Weights & Biases](https://wandb.ai):

```
WANDB_PROJECT=a2c-assignment
```

## Partes 1, 2 e 3 — Implementação

Complete os três blocos marcados com `YOUR CODE HERE`. **Atenção: eles estão em dois
arquivos diferentes.**

Em `algorithms/a2c.py`:

- **Parte 1** — `compute_n_step_returns`: o retorno de n passos com *bootstrap*,
  $R_t = r_t + \gamma\,(1 - d_t)\,R_{t+1}$, com $R_T = V(s_T)$. Percorra o rollout de
  trás para frente; um `done` interrompe a propagação. Cada coluna (ambiente) é
  independente.
- **Parte 2** — `compute_policy_loss`: a perda do gradiente de política,
  $-\frac{1}{B}\sum_i \log\pi(a_i|s_i)\,A_i$, com $A_i = R_i - V(s_i)$ (ou apenas $R_i$
  quando `use_baseline=False`). Pense com cuidado sobre *o que é constante* nessa
  expressão do ponto de vista do gradiente — um dos testes verifica exatamente isso.

Em `networks/discrete_actor_critic.py`:

- **Parte 3** — `DiscreteActorCritic.get_action_and_value`: dados os logits do ator,
  construa a distribuição categórica, amostre (ou tome o argmax se `deterministic`),
  e devolva ação, log-probabilidade, entropia e V(s) com as formas documentadas.
  Se uma ação for passada, avalie a log-probabilidade *dela* em vez de amostrar.

Cada bloco tem poucas linhas. Verifique com:

```bash
python -m pytest tests/test_a2c.py
```

Todos os testes devem passar antes de você começar a Parte 4. Depois treine (sempre a
partir da raiz do repositório):

```bash
python train.py --config a2c_cartpole
```

Os hiperparâmetros estão em [configs/a2c_cartpole.yml](../configs/a2c_cartpole.yml);
qualquer um deles pode ser alterado por run com `--override` (veja abaixo) — você não
deve precisar editar o arquivo de config.

Um treino completo no CartPole (500 mil passos, 8 ambientes) leva **poucos minutos** em
CPU de notebook — bem menos que o DQN — e deve chegar a retorno médio acima de 450, com a
avaliação final em 500 (o máximo). A curva do A2C é visivelmente mais *irregular* que a
do DQN: oscilações e quedas temporárias são normais; não aprender nunca, não. Como cada
run é barato, **rode 2 seeds** sempre que a sua conclusão depender de uma configuração.

Dica: se for rodar vários treinos ao mesmo tempo, limite as threads de cada um, senão
eles disputam CPU e todos ficam ~10× mais lentos:

```bash
OMP_NUM_THREADS=1 python train.py --config a2c_cartpole --override seed=2
```

## Os gráficos com que você vai trabalhar

Todo run registra estes gráficos no wandb; seu relatório deve se basear neles
(screenshots ou links de relatório do wandb, sempre com os runs identificados):

| Gráfico | O que ele diz |
|---|---|
| `charts/episodic_return_mean_last100` | a curva de aprendizado — retorno médio dos últimos 100 episódios |
| `losses/policy_loss` | $-\overline{\log\pi\cdot A}$ no lote — o *sinal* e o *ruído* deste gráfico importam mais que o valor |
| `losses/value_loss` | quão longe V(s) está do retorno de n passos |
| `losses/explained_variance` | quanto da variância dos retornos o crítico explica (1 = perfeito, ≤ 0 = inútil) |
| `losses/entropy` | entropia média da política ($\ln 2 \approx 0{,}69$ = uniforme no CartPole; 0 = determinística) |
| `charts/advantage_std` | desvio-padrão dos pesos $A_i$ usados no gradiente de política neste lote |
| `charts/advantage_mean` | média desses pesos |
| `losses/grad_norm` | norma do gradiente antes do clipping |
| `charts/SPS` | passos de ambiente por segundo |
| `eval/mean_return`, `eval/std_return` | avaliação final de 10 episódios |

Como no DQN: **nunca leia um gráfico isoladamente.** A curva de retorno diz *se* algo
deu errado; `entropy`, `explained_variance` e `advantage_std` costumam dizer *o quê*.

## Parte 4 — Relatório experimental

Para cada questão abaixo, siga o mesmo protocolo da atividade anterior:

1. **Preveja** (antes de rodar — no máximo 2–3 frases): o que você espera que os
   gráficos indicados façam ao longo da sua varredura, e por quê?
2. **Varredura (sweep)**: escolha **pelo menos dois valores** do hiperparâmetro (à sua
   escolha, cobrindo de pequeno → grande o bastante para expor o comportamento), além
   do baseline. Rode cada configuração; rode 2 seeds nas configurações das quais seu
   argumento depende (`--override seed=2`).
3. **Reporte os gráficos**: para cada questão, inclua os gráficos indicados, com todos
   os valores da varredura sobrepostos e os runs identificados.
4. **Explique**: 1–2 parágrafos sobre o *mecanismo* por trás do que você observou —
   "piorou" é uma descrição, não uma explicação. As explicações devem referenciar os
   gráficos e dizer se a sua previsão se confirmou.

Os overrides funcionam assim (qualquer chave de
[configs/a2c_cartpole.yml](../configs/a2c_cartpole.yml)):

```bash
python train.py --config a2c_cartpole --override a2c.num_steps=32 seed=2
```

**Q1 — Número de atores em paralelo** (`num_envs`; o baseline usa 8. Experimente 1 e
algo bem maior).

Gráficos a reportar: `charts/episodic_return_mean_last100`, `losses/policy_loss`,
`charts/SPS`.

Esse é o argumento central do A3C: por que vários atores ajudam um método *on-policy*?
Olhe o `policy_loss` com 1 ambiente — o que o ruído dele diz sobre o gradiente que
está sendo seguido? Atenção: o tamanho do batch é
`num_envs × num_steps`, então mudar `num_envs` também muda quantas transições entram em
cada atualização e quantas atualizações cabem em 500 mil passos. Separe os dois
efeitos na sua explicação. Compare também o `SPS`: o que o paralelismo compra em tempo total?

**Q2 — Horizonte do retorno de n passos** (`a2c.num_steps`; o baseline usa 5 — o
`t_max` do artigo do A3C. Experimente 1 e valores grandes, p. ex. 32 e 128).

Gráficos a reportar: `charts/episodic_return_mean_last100`, `losses/value_loss`,
`losses/explained_variance`.

Com `num_steps=1` o alvo é quase todo *bootstrap* ($r + \gamma V$); com `num_steps`
grande é quase Monte Carlo. Qual dos dois tem mais viés? Qual tem mais variância? O que
isso faz com o crítico (`value_loss`, `explained_variance`) — e por que o ator sofre
quando o crítico sofre? Cuidado com uma armadilha: com `num_steps=1` a
`explained_variance` pode parecer *ótima* — por que isso não significa que o crítico
seja bom? (Atenção: o tamanho do lote muda.)

**Q3 — Coeficiente de entropia** (`a2c.ent_coef`; o baseline usa 0,01. Experimente 0 e
algo grande, p. ex. 0,1. **Rode 2 seeds no `ent_coef=0`.**)

Gráficos a reportar: `charts/episodic_return_mean_last100`, `losses/entropy`.

O que o bônus de entropia impede? Com `ent_coef` grande, o que acontece com a entropia
e por que o retorno estaciona? Com `ent_coef=0` o CartPole *pode* funcionar — se
funcionou nos seus runs, explique por que esse ambiente é benevolente e o que você
esperaria num ambiente com mais ações ou recompensas mais raras.

**Q4 — Ablação do baseline** (`a2c.use_baseline=false`: o peso do gradiente de política
passa a ser o retorno $R$ em vez da vantagem $R - V(s)$. **Rode 2 seeds.**)

Gráficos a reportar: `charts/episodic_return_mean_last100`, `charts/advantage_std`,
`charts/advantage_mean`, `losses/entropy`.

Esse é o resultado teórico central da aula: subtrair um baseline não muda o gradiente
esperado, mas muda a variância. Confirme nos gráficos de `advantage_*`. Depois olhe a
entropia: no CartPole todas as recompensas são positivas — o que um peso *sempre
positivo* faz com a probabilidade de qualquer ação que tenha sido amostrada, e o que
isso tem a ver com a entropia colapsar mais rápido?

**Extra — LunarLander.** O CartPole é benevolente: recompensas densas, sempre
positivas, duas ações. Treine o seu A2C em `LunarLander-v3` (recompensas de −400 a
+250, com um grande termo terminal; considera-se resolvido com retorno médio ≥ 200) e
tente resolvê-lo. Não há config pronta: comece do `a2c_cartpole` e ajuste via
`--override` (`env_id=LunarLander-v3` e o que mais julgar necessário — quantos passos,
quantos ambientes, hiperparâmetros) ou crie o seu próprio `configs/a2c_lunarlander.yml`.
Reporte a curva de retorno e os gráficos que justificam as suas escolhas, e diga o que
mudou em relação ao CartPole e por quê. Chegar perto sem resolver também vale — desde
que você explique o que está limitando. (Fora da imagem Docker do curso, o LunarLander
requer `pip install swig && pip install --no-build-isolation box2d-py`.)

## Entregáveis

Um relatório em PDF (máx. 3 páginas) com as quatro seções tendo previsão → varredura →
gráficos → explicação, contendo um link para o seu fork com `algorithms/a2c.py` e
`networks/discrete_actor_critic.py` completos.
