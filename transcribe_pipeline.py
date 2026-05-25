import os
import glob
import warnings
import sys
import torch
import whisper
from pyannote.audio import Pipeline
from openai import OpenAI
from huggingface_hub import login
import huggingface_hub.utils._validators as validators

warnings.filterwarnings("ignore")

# ── Patch de compatibilidade pyannote ─────────────────────────────────────────
def smoothly_deprecate_legacy_arguments(fn_name, kwargs):
    if "use_auth_token" in kwargs:
        kwargs.pop("use_auth_token")
    return kwargs
validators.smoothly_deprecate_legacy_arguments = smoothly_deprecate_legacy_arguments

# ── Configuração via variáveis de ambiente ────────────────────────────────────
# Nunca coloque chaves diretamente no código.
# Configure as variáveis abaixo antes de rodar:
#   export OPENAI_API_KEY="sk-..."
#   export HF_TOKEN="hf_..."
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
HF_TOKEN       = os.environ["HF_TOKEN"]

# ── Pastas ────────────────────────────────────────────────────────────────────
# Ajuste os caminhos conforme seu ambiente (local ou Google Drive)
PASTA_GRAVACOES   = os.environ.get("PASTA_GRAVACOES",   "./audios")
PASTA_TRANSCRICOES = os.environ.get("PASTA_TRANSCRICOES", "./output")
os.makedirs(PASTA_TRANSCRICOES, exist_ok=True)

# ── Clientes ──────────────────────────────────────────────────────────────────
client_chatgpt = OpenAI(api_key=OPENAI_API_KEY)
login(token=HF_TOKEN)

# ── Modelos ───────────────────────────────────────────────────────────────────
print("🚀 Carregando modelos...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

pipeline_diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1"
).to(device)

model_whisper = whisper.load_model("medium", device=device)

# ── Prompt de análise de QA ───────────────────────────────────────────────────
QA_PROMPT_TEMPLATE = """
Você é especialista em análise de qualidade de vendas. Analise a transcrição abaixo
e responda EXATAMENTE no formato especificado.

{transcricao}

Responda neste formato:

1️⃣ Cliente já possuía o serviço?: [Sim/Não — cliente tinha o serviço contratado antes desta ligação?]
2️⃣ Indução de endereço?: [Sim/Não — vendedor sugeriu usar endereço/complemento diferente do real para contornar impedimento técnico ou cadastral?]
3️⃣ Data de instalação informada?: [Sim/Não — vendedor comunicou ao cliente a data prevista de instalação?]
4️⃣ Produto vendido: [Internet / TV / Ambos]
5️⃣ Dados do cadastro: [NOME COMPLETO, CPF, NOME DA MÃE, ENDEREÇO COMPLETO — informe "NÃO ENCONTRADO" para campos ausentes]
6️⃣ Trecho de indução: [Trecho exato onde ocorreu indução, ou "NENHUMA"]
"""

# ── Pipeline principal ────────────────────────────────────────────────────────
arquivos_mp3 = glob.glob(os.path.join(PASTA_GRAVACOES, "*.mp3"))
print(f"✅ {len(arquivos_mp3)} arquivo(s) encontrado(s)\n")

estatisticas = {
    "total": len(arquivos_mp3),
    "sim": 0,
    "nao": 0,
    "parcial": 0,
    "indefinido": 0,
}

for audio_file in arquivos_mp3:
    nome_base      = os.path.splitext(os.path.basename(audio_file))[0]
    partes         = nome_base.split("-")
    identificador  = f"{partes[0]}_{partes[2]}" if len(partes) >= 3 else nome_base

    print(f"🔄 Processando: {identificador}")

    # 1. Diarização (identifica speakers)
    diarization = pipeline_diarization(audio_file)

    # 2. Transcrição com timestamps por palavra
    resultado_whisper = model_whisper.transcribe(
        audio_file,
        word_timestamps=True,
        language="pt"
    )

    # 3. Alinha transcrição com speakers
    linhas = []
    for seg in resultado_whisper["segments"]:
        texto   = seg["text"].strip()
        t_start = seg["start"]
        t_end   = seg["end"]
        speaker = "?"

        for diar_seg, _, label in diarization.itertracks(yield_label=True):
            if diar_seg.start <= t_start < diar_seg.end:
                speaker = label
                break

        if texto and speaker != "?":
            linhas.append(f"[{t_start:.1f}s-{t_end:.1f}s] *{speaker}*: {texto}")

    transcricao = "\n".join(linhas)

    # 4. Análise de QA com GPT
    print("  🤖 Analisando com GPT...")
    resposta = client_chatgpt.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": QA_PROMPT_TEMPLATE.format(transcricao=transcricao)
        }]
    )
    analise = resposta.choices[0].message.content

    # 5. Classifica resultado da pergunta de indução (item 2)
    texto_upper = analise.upper()
    linha_inducao = next(
        (ln for ln in texto_upper.splitlines() if "2️⃣" in ln or "INDU" in ln),
        texto_upper
    )
    conteudo = (
        linha_inducao.split("[", 1)[1].split("]", 1)[0].strip()
        if "[" in linha_inducao else linha_inducao
    )
    primeira = conteudo.split()[0] if conteudo else ""

    if primeira.startswith("SIM"):
        status = "✅ SIM"
        estatisticas["sim"] += 1
    elif "PARCIAL" in conteudo:
        status = "⚠️ PARCIAL"
        estatisticas["parcial"] += 1
    elif primeira.startswith("NÃO") or primeira.startswith("NAO"):
        status = "❌ NÃO"
        estatisticas["nao"] += 1
    else:
        status = "❓ INDEFINIDO"
        estatisticas["indefinido"] += 1

    # 6. Salva resultado
    caminho_saida = os.path.join(PASTA_TRANSCRICOES, f"{identificador}.txt")
    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write("=== TRANSCRIÇÃO ===\n\n")
        f.write(transcricao)
        f.write(f"\n\n{'='*60}\nANÁLISE DE QUALIDADE\n{'='*60}\n\n{analise}")

    print(f"  {status} → salvo em {identificador}.txt\n")

# ── Resumo final ──────────────────────────────────────────────────────────────
print("=" * 50)
print(f"✅ Concluído! {estatisticas['total']} arquivo(s) processado(s)")
print(f"   SIM      : {estatisticas['sim']}")
print(f"   NÃO      : {estatisticas['nao']}")
print(f"   PARCIAL  : {estatisticas['parcial']}")
print(f"   INDEFINIDO: {estatisticas['indefinido']}")
print(f"\nResultados salvos em: {PASTA_TRANSCRICOES}")
