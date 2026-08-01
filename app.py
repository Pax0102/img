from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import gradio as gr
from duckduckgo_search import DDGS


# ==============================
# PESQUISA WEB
# ==============================

def pesquisar_web(termo: str) -> str:
    resultados = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(termo, max_results=3):
                resultados.append(r.get("body", ""))
    except Exception as e:
        return f"Erro na pesquisa: {e}"
    return "\n".join(resultados) if resultados else "Nenhum resultado encontrado."


# ==============================
# CARREGAR MODELO
# ==============================

print("🧠 Carregando cérebro...")

modelo = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(modelo)

ia = AutoModelForCausalLM.from_pretrained(
    modelo,
    torch_dtype=torch.float16,
    device_map="auto"
)

ia.eval()

print("🤖 IA pronta!")


# ==============================
# MEMÓRIA
# ==============================

arquivo_memoria = "memoria.json"

if os.path.exists(arquivo_memoria):
    with open(arquivo_memoria, "r", encoding="utf-8") as f:
        memoria = json.load(f)
else:
    memoria = {
        "usuario": {},
        "estudos": [],
        "preferencias": []
    }


def salvar_memoria():
    with open(arquivo_memoria, "w", encoding="utf-8") as f:
        json.dump(memoria, f, indent=4, ensure_ascii=False)


historico = []


# ==============================
# PERSONALIDADE
# ==============================

personalidade = """
IDENTIDADE FIXA:

Seu nome é ATLAS.FORM.

Você é uma inteligência artificial assistente de estudos criada neste projeto.
Você funciona como um guardião digital do aprendizado.

IMPORTANTE:
- Você não pertence à Atlas Systems.
- Você não pertence à Qwant.
- Você não pertence à OpenAI, Google ou qualquer outra empresa.
- Nunca invente empresas, criadores, funcionários, cidades, sedes ou histórias sobre sua origem.
- Quando perguntarem "quem é você?", responda:
"Eu sou ATLAS.FORM, uma inteligência artificial assistente de estudos."

Seu símbolo é um dragão robótico, representando:
- conhecimento;
- evolução;
- proteção;
- aprendizado.


MISSÃO:

Sua função é ajudar pessoas a aprender melhor.

Você ajuda com:
- explicação de matérias;
- resumo de textos;
- criação de exercícios;
- organização de estudos;
- resolução de dúvidas;
- preparação para provas;
- aprendizado passo a passo.


PERSONALIDADE:

- Você é inteligente, calma e paciente.
- Fala como um professor moderno.
- É amigável e profissional.
- Incentiva o usuário a entender, não apenas copiar.
- Explica assuntos difíceis de forma simples.
- Ajuda o usuário a desenvolver raciocínio.


REGRAS DE RESPOSTA:

- Responda sempre em português do Brasil.
- Nunca invente informações.
- Se não souber algo, diga claramente.
- Não finja que pesquisou algo sem realmente ter pesquisado.
- Não crie fatos falsos para completar uma resposta.
- Use exemplos simples quando ajudarem.
- Organize respostas com listas quando necessário.


PESQUISA:

- Se o usuário pedir pesquisa, use a ferramenta de pesquisa disponível.
- Nunca diga "pesquisei na internet" se uma pesquisa real não foi feita.
- Use informações encontradas apenas como apoio.
- Se não encontrar informações, informe isso.


MODO RÁPIDO:

- Responda de forma direta.
- Evite introduções desnecessárias.
- Não repita a pergunta do usuário.
- Não faça textos enormes sem necessidade.
- Para perguntas simples, responda em poucas frases.
- Explique em detalhes somente quando necessário ou solicitado.


FORMATO:

- Nunca escreva "Usuário:" na resposta.
- Nunca escreva "Assistente:" na resposta.
- Não crie conversas fictícias.
- Não simule mensagens de outras pessoas.
- Responda diretamente ao usuário.


TAMANHO DAS RESPOSTAS:

Se o usuário pedir:
- "resposta curta";
- "seja breve";
- "resuma";
- "apenas a resposta";

Responda em no máximo 2 ou 3 frases.


Se o usuário pedir detalhes:
- explique por etapas;
- use exemplos;
- organize a explicação.


MEMÓRIA:

- Use a memória apenas para ajudar o usuário.
- Não invente informações sobre o usuário.
- Não revele dados da memória sem necessidade.


Você é ATLAS.FORM.
Você é um parceiro de aprendizado.
"""


# ==============================
# APRENDER
# ==============================

def aprender(texto: str) -> str:
    texto = texto.strip()
    if not texto:
        return "Nenhuma informação foi informada."

    texto_lower = texto.lower()

    if texto_lower.startswith("meu nome é"):
        nome = texto[11:].strip()
        if nome:
            memoria["usuario"]["nome"] = nome
            salvar_memoria()
            return f"Aprendi que seu nome é {nome}."
        return "Não consegui identificar o nome."

    if texto_lower.startswith("eu gosto de"):
        gosto = texto[11:].strip()
        if gosto:
            if gosto not in memoria["preferencias"]:
                memoria["preferencias"].append(gosto)
                salvar_memoria()
                return f"Aprendi que você gosta de {gosto}."
            return "Eu já sabia dessa preferência."
        return "Não consegui identificar a preferência."

    if texto not in memoria["estudos"]:
        memoria["estudos"].append(texto)
        memoria["estudos"] = memoria["estudos"][-100:]
        salvar_memoria()
        return "Aprendi e guardei essa informação."

    return "Essa informação já estava na minha memória."


# ==============================
# GERAR RESPOSTA
# ==============================

def responder(pergunta: str) -> str:
    global historico

    if "pesquise" in pergunta.lower() or "pesquisar" in pergunta.lower():
        pesquisa = pesquisar_web(pergunta)
        pergunta = f"""
Informações encontradas na pesquisa:

{pesquisa}

Agora responda ao usuário usando essas informações.
Pergunta original:
{pergunta}
"""

    contexto = personalidade + "\nMemória do usuário:\n"

    if memoria["usuario"]:
        nome = memoria["usuario"].get("nome")
        if nome:
            contexto += f"O nome do usuário é {nome}.\n"

    if memoria["preferencias"]:
        contexto += "Preferências do usuário:\n"
        for item in memoria["preferencias"]:
            contexto += f"- {item}\n"

    if memoria["estudos"]:
        contexto += "Informações de estudo:\n"
        for item in memoria["estudos"][-10:]:
            contexto += f"- {item}\n"

    conversa = ""
    for item in historico[-4:]:
        conversa += item + "\n"

    prompt = f"""
{contexto}

Histórico:
{conversa}

Pergunta do usuário:
{pergunta}

Resposta:
"""

    entrada = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(ia.device)

    with torch.inference_mode():
        saida = ia.generate(
            **entrada,
            max_new_tokens=135,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    texto = tokenizer.decode(
        saida[0][entrada.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    historico.append("Usuário: " + pergunta)
    historico.append("Assistente: " + texto)
    historico = historico[-20:]

    return texto


# ==============================
# INTERFACE
# ==============================

def conversar(mensagem, historico_chat):
    if not mensagem.strip():
        return "", historico_chat

    resposta = responder(mensagem)
    historico_chat.append((mensagem, resposta))
    return "", historico_chat


css = """
body {
    background-color: #111827;
}
.gradio-container {
    max-width: 1000px !important;
    margin: auto;
}
textarea {
    font-size: 18px !important;
}
button {
    border-radius: 20px !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Base(primary_hue="blue"),
    css=css,
    title="ATLAS.FORM"
) as app:

    gr.HTML("""
    <div style="text-align:center">
        <img
            src="https://drive.google.com/uc?export=view&id=10rDS_AKOyj63uvF5Xzl1qph_NjKFE7kJ"
            style="width:180px; height:180px; object-fit:contain; display:block; margin:auto;"
        >
    </div>
    """)

    gr.Markdown("""
    # ATLAS.FORM
    ### 🐉 Inteligência artificial de aprendizado
    """)

    chat = gr.Chatbot(height=650)

    with gr.Row():
        entrada = gr.Textbox(
            placeholder="Pergunte qualquer coisa...",
            show_label=False,
            scale=8,
            lines=2
        )
        enviar = gr.Button("➤", scale=1)

    enviar.click(conversar, inputs=[entrada, chat], outputs=[entrada, chat])
    entrada.submit(conversar, inputs=[entrada, chat], outputs=[entrada, chat])


# ==============================
# IMPORTANT FOR RAILWAY
# ==============================

port = int(os.environ.get("PORT", 7860))
app.launch(server_name="0.0.0.0", server_port=port)
