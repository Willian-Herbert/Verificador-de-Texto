from flask import Flask, request, render_template, send_file, redirect, url_for, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from fpdf import FPDF
from collections import Counter
from nltk.corpus import stopwords
import nltk
import io
import numpy as np
import tempfile
import os
import uuid
import re
import json

stopwords = nltk.corpus.stopwords.words('portuguese')

app = Flask(__name__)
app.secret_key = 'chave_secreta_para_sessao'

def extrair_texto_pdf(arquivo):
    leitor = PdfReader(arquivo)
    texto = ""
    for pagina in leitor.pages:
        texto += pagina.extract_text() or ""
    return texto

def limpar_texto(texto):
    texto = re.sub(r'\s+', ' ', texto)  # remove quebras de linha e espaços extras
    texto = re.sub(r'[^a-zA-Z0-9À-ÿ\s]', '', texto)  # remove caracteres especiais
    return texto.strip()

def analisar_texto_ia(texto):
    # Tokenização simples por espaço
    palavras = texto.lower().split()

    # Remover stopwords
    palavras_filtradas = [
        palavra for palavra in palavras if palavra not in stopwords
    ]

    total_palavras = len(palavras_filtradas)
    palavras_unicas = len(set(palavras_filtradas))

    if total_palavras == 0:
        return {
            "diversidade_lexical": 0,
            "mensagem": "⚠️ Texto vazio após remoção de stopwords ou não processável."
        }

    diversidade_lexical = palavras_unicas / total_palavras
    contagem = Counter(palavras_filtradas)
    palavra_mais_comum, freq_mais_comum = contagem.most_common(1)[0]

    mensagem = "✔️ Texto com boa diversidade lexical."

    if diversidade_lexical < 0.3:
        mensagem = (
            "⚠️ Baixa diversidade lexical. Isso pode indicar texto gerado automaticamente ou com alta repetição."
        )

    return {
        "diversidade_lexical": round(diversidade_lexical, 3),
        "palavra_mais_comum": palavra_mais_comum,
        "frequencia_mais_comum": freq_mais_comum,
        "mensagem": mensagem
    }

def calcular_similaridade(texto1, texto2):
    if not texto1.strip() or not texto2.strip():
        return 0.0
    vetor = TfidfVectorizer(stop_words=stopwords).fit_transform([texto1, texto2])
    if vetor.shape[1] == 0:
        return 0.0
    similaridade = cosine_similarity(vetor[0:1], vetor[1:2])
    # print(f"Similaridade calculada: {TfidfVectorizer(stop_words=stopwords).fit_transform([texto1])}") Tentar entender esse resultado de vetorização
    return float(similaridade[0][0])

def encontrar_trechos_similares(texto1, texto2, top_n=3, limite_caracteres=400, bloco_palavras=50):
    def dividir_em_blocos(texto, tamanho):
        palavras = texto.split()
        return [
            ' '.join(palavras[i:i + tamanho])
            for i in range(0, len(palavras), tamanho)
            if len(palavras[i:i + tamanho]) > 0
        ]

    blocos1 = dividir_em_blocos(texto1, bloco_palavras)
    blocos2 = dividir_em_blocos(texto2, bloco_palavras)
    pares_similares = []

    for s1 in blocos1:
        for s2 in blocos2:
            vetor = TfidfVectorizer(stop_words=stopwords).fit_transform([s1, s2])
            if vetor.shape[1] == 0:
                continue
            sim = cosine_similarity(vetor[0:1], vetor[1:2])[0][0]
            pares_similares.append((sim, s1, s2))

    pares_similares.sort(reverse=True, key=lambda x: x[0])

    # Garante que sempre retorna até top_n trechos
    while len(pares_similares) < top_n:
        pares_similares.append((0.0, "Sem trecho suficiente", "Sem trecho suficiente"))

    trechos_truncados = []
    for sim, s1, s2 in pares_similares[:top_n]:
        s1_truncado = (s1[:limite_caracteres] + '...') if len(s1) > limite_caracteres else s1
        s2_truncado = (s2[:limite_caracteres] + '...') if len(s2) > limite_caracteres else s2
        trechos_truncados.append((sim, s1_truncado, s2_truncado))

    return trechos_truncados

def gerar_pdf(resultado, trechos):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Comparação de Textos", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Índice de Correlação: {resultado * 100:.2f}%", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Trechos mais similares:", ln=True)

    for i, (sim, s1, s2) in enumerate(trechos, 1):
        pdf.multi_cell(0, 10, txt=f"Trecho {i} - Similaridade: {sim * 100:.2f}%\nTexto 1: {s1}\nTexto 2: {s2}\n")
        pdf.ln(5)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    temp_file.close()
    return temp_file.name

def salvar_em_arquivo_temporario(conteudo, suffix='.txt'):
    caminho = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")
    with open(caminho, 'w', encoding='utf-8') as f:
        if suffix == '.json':
            json.dump(conteudo, f)
        else:
            f.write(conteudo)
    return caminho

def carregar_texto_do_arquivo(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        return f.read()

def carregar_json_do_arquivo(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    trechos = None
    analise_ia1 = None
    analise_ia2 = None
    if request.method == 'POST':
        arquivo1 = request.files['arquivo1']
        arquivo2 = request.files['arquivo2']

        if arquivo1.filename.endswith('.pdf'):
            texto1 = extrair_texto_pdf(arquivo1)
        else:
            texto1 = arquivo1.read().decode('utf-8')

        if arquivo2.filename.endswith('.pdf'):
            texto2 = extrair_texto_pdf(arquivo2)
        else:
            texto2 = arquivo2.read().decode('utf-8')

        texto1 = limpar_texto(texto1)
        texto2 = limpar_texto(texto2)

        resultado = calcular_similaridade(texto1, texto2)
        trechos = encontrar_trechos_similares(texto1, texto2)

        analise_ia1 = analisar_texto_ia(texto1)
        analise_ia2 = analisar_texto_ia(texto2)

        session['resultado'] = resultado
        session['texto1_path'] = salvar_em_arquivo_temporario(texto1)
        session['texto2_path'] = salvar_em_arquivo_temporario(texto2)
        session['trechos_path'] = salvar_em_arquivo_temporario(trechos, suffix='.json')
        session['analise_ia1'] = analise_ia1
        session['analise_ia2'] = analise_ia2

        return redirect(url_for('index'))

    resultado = session.get('resultado')
    trechos_path = session.get('trechos_path')
    trechos = carregar_json_do_arquivo(trechos_path) if trechos_path else None
    analise_ia1 = session.get('analise_ia1', {})
    analise_ia2 = session.get('analise_ia2', {})
    
    return render_template(
    'index.html',
    resultado=resultado,
    trechos=trechos,
    analise_ia1=analise_ia1,
    analise_ia2=analise_ia2
    )

@app.route('/baixar-relatorio')
def baixar_relatorio():
    resultado = session.get('resultado')
    texto1 = carregar_texto_do_arquivo(session.get('texto1_path', ''))
    texto2 = carregar_texto_do_arquivo(session.get('texto2_path', ''))
    trechos_path = session.get('trechos_path')
    trechos = carregar_json_do_arquivo(trechos_path) if trechos_path else []
    caminho_pdf = gerar_pdf(resultado, trechos)
    return send_file(caminho_pdf, as_attachment=True, download_name='relatorio_comparacao.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
