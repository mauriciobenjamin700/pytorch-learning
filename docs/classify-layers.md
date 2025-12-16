# 🧠 MODELO MENTAL: CADA CAMADA É UM CONTRATO

Para **toda camada**, você deve saber:

1. **O que ela recebe**
2. **De quem ela recebe**
3. **O que ela faz**
4. **O que ela devolve**
5. **Para quem ela devolve**
6. **O que NÃO é responsabilidade dela**

Vamos passar camada por camada **numa CNN de classificação**, sem pular nada.

---

## 📥 1. ENTRADA (Input)

### ✔ ENTRADA Recebe

* Um tensor: `torch.Tensor`
* Shape: `(batch, channels, height, width)`
  * Ex: `(32, 3, 32, 32)`

#### ENTRADA Legenda

| Dimensão | Significado               |
| -------- | ------------------------- |
| `batch`  | Quantas imagens           |
| `channels`   | Quantos canais de entrada |
| `H`      | Altura                    |
| `W`      | Largura                   |

Imagem RGB → C_in = 3

Imagem em escala de cinza → C_in = 1

Feature map intermediário → C_in = nº de filtros anteriores

### ✔ENTRADA Quem envia

* `DataLoader`

### ❌ ENTRADA Não faz

* Não aprende
* Não normaliza
* Não classifica

---

## 🔲 2. Conv2d — *Extrator de padrões locais*

### ✔ Conv2d Recebe

```python
(batch, C_in, H, W)
```

#### Conv2d Legenda

| Dimensão | Significado               |
| -------- | ------------------------- |
| `batch`  | Quantas imagens           |
| `C_in`   | Quantos canais de entrada |
| `H`      | Altura                    |
| `W`      | Largura                   |

### ✔ Conv2d Recebe de

* Entrada da rede **ou**
* Saída de outra Conv2d

### ✔ Conv2d Faz

* Aplica filtros convolucionais
* Detecta padrões **locais**
* Aprende pesos (kernels)

### ✔ Conv2d Retorna

```python
(batch, C_out, H_out, W_out)
```

| Dimensão | Significado             |
| -------- | ----------------------- |
| `batch`  | Mesmo lote              |
| `C_out`  | Nº de filtros da Conv   |
| `H_out`  | Altura após convolução  |
| `W_out`  | Largura após convolução |

Cada filtro gera 1 mapa de ativação

C_out = quantos padrões diferentes a camada aprende

```bash
nn.Conv2d(3, 32, 3)
```

```bash
→ C_out = 32
```

#### Conv2d Legenda Retorno

| Dimensão | Significado               |
| -------- | ------------------------- |
| `batch`  | Quantas imagens           |
| `C_out`   | Quantos canais de entrada |
| `H`      | Altura                    |
| `W`      | Largura                   |

### ✔ Conv2d Retorna para

* Ativação (ReLU)
* Normalização
* Outra Conv

### ❌ Conv2d NÃO faz

* Decidir classe
* Reduzir batch
* Garantir não-linearidade

📌 **Regra mental**:

> Conv só se preocupa com **características espaciais**

---

## ⚡ 3. ReLU — *Decisor de ativação*

### ✔ ReLU Recebe

```python
(batch, C, H, W)
```

### ✔ ReLU Recebe de

* Conv
* Linear

### ✔ ReLU Faz

* Zera valores negativos
* Introduz **não-linearidade**

### ✔ ReLU Retorna

```python
(batch, C, H, W)
```

### ✔ ReLU Retorna para

* Pooling
* Outra Conv
* Linear

### ❌ ReLU NÃO faz

* Aprender pesos
* Extrair padrões
* Normalizar dados

📌 **Regra mental**:

> ReLU só decide **o que passa e o que morre**

---

## 📉 4. Pooling — *Redutor de resolução*

### ✔ Pooling Recebe

```python
(batch, C, H, W)
```

### ✔ Pooling Recebe de

* ReLU

### ✔ Pooling Faz

* Reduz resolução espacial
* Mantém informação dominante

### ✔ Pooling Retorna

```python
(batch, C, H/2, W/2)
```

| Dimensão | Mudança          |
| -------- | ---------------- |
| `batch`  | Igual            |
| `C`      | Igual            |
| `H/2`    | Altura reduzida  |
| `W/2`    | Largura reduzida |

Não cria novos padrões

Só reduz a resolução

### ✔ Pooling Retorna para

* Outra Conv

### ❌ Pooling NÃO faz

* Aprender
* Classificar
* Mudar número de canais

📌 **Regra mental**:

> Pool reduz *onde* a informação está, não *o que* ela é

---

## 🔁 BLOCO COMPLETO (conv → relu → pool)

👉 **Responsabilidade do bloco**:

> “Transformar uma imagem mais simples numa representação mais abstrata e menor”

---

## 🧱 5. Stack de Convs — *Hierarquia de significado*

### Fluxo real

```bash
Conv (bordas)
→ Conv (texturas)
→ Conv (partes)
→ Conv (objetos)
```

Cada Conv **confia** que:

* a anterior já organizou a informação

---

## 📐 6. Flatten — *Mudança de contrato*

### ✔ Flatten Recebe

```python
(batch, C, H, W)
```

### ✔ Flatten Recebe de

* Última Conv

### ✔ Flatten Faz

* Converte tensor espacial em vetor

### ✔ Flatten Retorna

```python
(batch, C*H*W)
```

A imagem “vira uma lista de números”.

Antes

```bash
(batch, 64, 8, 8)
```

Depois

```bash
(batch, 4096)
```

👉 Aqui acaba a noção de espaço (H e W somem).

### ✔ Flatten Retorna para

* Linear

### ❌ Flatten NÃO faz

* Aprender
* Normalizar
* Classificar

📌 **Aqui muda tudo**:
👉 a rede deixa de ser espacial e vira **vetorial**

---

## 🧮 7. Linear — *Combinador global*

### ✔ Linear Recebe

```python
(batch, features)
```

Features = “informações já extraídas”

Não é mais imagem

É um vetor

### ✔ Linear Recebe de

* Flatten
* Outra Linear

### ✔ Linear Faz

* Combina TODAS as features
* Aprende relações globais

### ✔ Linear Retorna

```python
(batch, output_features)
```

Saída de uma camada Linear intermediária.

Representação abstrata

Combinação global das features

```bash
nn.Linear(4096, 128)
```

```bash
→ hidden_features = 128
```

### ✔ Linear Retorna para

* Outra Linear
* Saída

### ❌ Linear NÃO faz

* Ver espaço
* Detectar padrões locais

📌 **Regra mental**:

> Linear só entende números, não imagens

---

## 🎯 8. Camada de saída — *Geradora de logits*

### ✔ Saída Recebe

```python
(batch, hidden_features)
```

Representação abstrata

Combinação global das features

```bash
nn.Linear(4096, 128)
→ hidden_features = 128
```

### ✔ Saída Recebe de

* Última Linear

### ✔ Saída Faz

* Gera um score por classe

### ✔ Saída Retorna

```python
(batch, num_classes)
```

### ✔ Saída Retorna para

* Função de loss

### ❌ Saída NÃO faz

* Softmax
* Decisão final

---

## 🧠 9. Loss — *Juiz*

### ✔ Loss Recebe

* logits `(batch, num_classes)`
* labels `(batch)`

### ✔ Loss Faz

* Mede erro
* Gera gradientes

### ❌ Loss NÃO faz

* Atualizar pesos

---

## 🔄 10. Optimizer — *Executor*

### ✔ Optimizer Recebe

* Gradientes

### ✔ Optimizer Faz

* Atualiza pesos

---

## 🔗 VISÃO COMPLETA DO FLUXO

```bash
DataLoader
 ↓
Conv → ReLU → Pool
 ↓
Conv → ReLU → Pool
 ↓
Flatten
 ↓
Linear → ReLU
 ↓
Linear (logits)
 ↓
Loss
 ↓
Backward
 ↓
Optimizer
```

🔗 VISÃO FINAL ENCADEADA

```bash
(batch, 3, H, W)          → imagem
↓
(batch, C1, H, W)        → conv
↓
(batch, C1, H/2, W/2)    → pool
↓
(batch, C2, H/2, W/2)    → conv
↓
(batch, C2, H/4, W/4)    → pool
↓
(batch, C2*H*W)          → flatten
↓
(batch, hidden_features) → linear
↓
(batch, num_classes)     → logits
```

---

## 🧠 REGRA DE OURO (GUARDE ISSO)

> **Cada camada só conhece o tipo de dado que recebe.
> Ela NÃO sabe o que veio antes, nem para onde vai depois.**

## 🧠 FRASE PARA GRAVAR

> Enquanto existir H e W, a rede está “olhando imagens”.
> Quando vira vetor, ela está “tomando decisões”.

---

## 🧪 EXERCÍCIO (ESSENCIAL)

Responda em voz alta ou escrevendo:

1️⃣ O que a `Conv2d` espera receber?
2️⃣ O que quebra se você mandar um vetor para uma Conv?
3️⃣ Por que `Flatten` só aparece uma vez?
4️⃣ Quem decide a classe? A rede ou a loss?
