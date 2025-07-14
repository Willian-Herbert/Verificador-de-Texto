from flask import Flask, request, render_template, send_file, redirect, url_for, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from fpdf import FPDF
from collections import Counter
import nltk
import tempfile
import os
import uuid
import re
import json
import math
import statistics
from textstat import flesch_reading_ease
import pickle
from pathlib import Path
from datetime import datetime
import sys

def log_step(message, details="", progress=None, step=None, total_steps=None):
    """Função para logging com timestamp e progresso detalhado"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Formato HH:MM:SS.mmm
    
    # Construir string de progresso
    progress_str = ""
    if progress is not None:
        progress_str = f" [{progress:5.1f}%]"
    
    if step is not None and total_steps is not None:
        remaining = total_steps - step
        progress_str += f" [Etapa {step}/{total_steps} - {remaining} restantes]"
    
    # Formatação da linha de log
    if details:
        print(f"[{timestamp}]{progress_str} {message} - {details}")
    else:
        print(f"[{timestamp}]{progress_str} {message}")
    
    # Forçar flush para aparecer imediatamente
    sys.stdout.flush()

class ProcessTracker:
    """Classe para rastrear progresso de processos complexos"""
    def __init__(self, name, total_steps):
        self.name = name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        self.substeps = {}
        
    def next_step(self, message, details=""):
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        remaining = self.total_steps - self.current_step
        
        log_step(
            f"🔄 {self.name}: {message}",
            details,
            progress=progress,
            step=self.current_step,
            total_steps=self.total_steps
        )
        
        return progress
    
    def substep(self, parent_step, message, current, total, details=""):
        """Para processos que têm sub-etapas (como processar páginas de PDF)"""
        substep_progress = (current / total) * 100 if total > 0 else 0
        overall_progress = ((self.current_step - 1 + current/total) / self.total_steps) * 100
        
        log_step(
            f"  ↳ {message}",
            f"{details} | Sub-progresso: {substep_progress:.1f}%",
            progress=overall_progress,
            step=current,
            total_steps=total
        )
    
    def complete(self, final_message=""):
        duration = (datetime.now() - self.start_time).total_seconds()
        log_step(
            f"✅ {self.name} CONCLUÍDO",
            f"{final_message} | Tempo total: {duration:.2f}s",
            progress=100.0
        )

# Inicialização única dos stopwords - português e inglês
try:
    STOPWORDS = set(nltk.corpus.stopwords.words('portuguese'))
    STOPWORDS.update(nltk.corpus.stopwords.words('english'))
except:
    nltk.download('stopwords')
    STOPWORDS = set(nltk.corpus.stopwords.words('portuguese'))
    STOPWORDS.update(nltk.corpus.stopwords.words('english'))

# Regex compilado para melhor performance
TEXTO_REGEX = re.compile(r'\s+')
CHAR_REGEX = re.compile(r'[^a-zA-ZÀ-ÿ\s]')  # Removido 0-9 para tirar números

app = Flask(__name__)
app.secret_key = 'chave_secreta_para_sessao'

# Diretório para arquivos temporários
TEMP_DIR = Path(tempfile.gettempdir()) / "ai_text_analyzer"
TEMP_DIR.mkdir(exist_ok=True)

# TfidfVectorizer reutilizável
vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS))

def extrair_texto_pdf(arquivo):
    """Extrai texto de PDF de forma otimizada com progresso detalhado"""
    tracker = ProcessTracker("EXTRAÇÃO PDF", 4)
    
    tracker.next_step("Carregando PDF", f"Arquivo: {arquivo.filename}")
    leitor = PdfReader(arquivo)
    total_paginas = len(leitor.pages)
    
    tracker.next_step("Analisando estrutura", f"{total_paginas} páginas encontradas")
    
    tracker.next_step("Extraindo texto das páginas", "Iniciando extração sequencial")
    texto_completo = ""
    
    for i, pagina in enumerate(leitor.pages, 1):
        texto_pagina = pagina.extract_text() or ""
        texto_completo += texto_pagina
        
        # Log detalhado do progresso
        tracker.substep(3, f"Página {i}", i, total_paginas, f"{len(texto_pagina)} caracteres extraídos")
    
    caracteres = len(texto_completo)
    tracker.next_step("Finalizando extração", f"{caracteres} caracteres totais extraídos")
    
    tracker.complete(f"PDF processado: {total_paginas} páginas, {caracteres} caracteres")
    return texto_completo

def limpar_texto(texto):
    """Limpa texto com progresso detalhado"""
    tracker = ProcessTracker("LIMPEZA DE TEXTO", 3)
    
    original_len = len(texto)
    tracker.next_step("Aplicando regex de espaços", f"Texto original: {original_len:,} caracteres")
    
    # Remover quebras de linha e espaços extras
    texto_limpo = TEXTO_REGEX.sub(' ', texto)
    after_spaces = len(texto_limpo)
    spaces_removed = original_len - after_spaces
    
    tracker.next_step("Removendo caracteres especiais", f"Após espaços: {after_spaces:,} caracteres (-{spaces_removed:,})")
    
    # Remover caracteres especiais
    texto_limpo = CHAR_REGEX.sub('', texto_limpo)
    texto_final = texto_limpo.strip()
    final_len = len(texto_final)
    chars_removed = after_spaces - final_len
    total_removed = original_len - final_len
    
    tracker.next_step("Finalizando limpeza", f"Caracteres especiais removidos: {chars_removed:,}")
    
    tracker.complete(f"Limpeza finalizada: {final_len:,} caracteres finais (-{total_removed:,} total)")
    return texto_final

def analisar_texto_ia(texto):
    """Análise avançada para detectar IA com progresso super detalhado"""
    tracker = ProcessTracker("ANÁLISE DE IA", 9)
    
    # Etapa 1: Tokenização
    tracker.next_step("Tokenizando texto", f"Analisando {len(texto):,} caracteres")
    palavras_completas = texto.lower().split()
    total_palavras_completas = len(palavras_completas)
    
    # Etapa 2: Filtrar stopwords
    tracker.next_step("Filtrando stopwords", f"{total_palavras_completas:,} palavras encontradas")
    palavras_filtradas = [p for p in palavras_completas if p not in STOPWORDS]
    total_palavras_filtradas = len(palavras_filtradas)
    stopwords_removidas = total_palavras_completas - total_palavras_filtradas
    
    if total_palavras_filtradas == 0:
        tracker.complete("ERRO: Texto vazio após filtros")
        return {
            "diversidade_lexical": 0,
            "probabilidade_ia": 0,
            "confianca": "Baixa",
            "detalhes": {},
            "mensagem": "⚠️ Texto vazio após remoção de stopwords ou não processável."
        }

    # Etapa 3: Métricas básicas
    tracker.next_step("Calculando diversidade lexical", f"{stopwords_removidas:,} stopwords removidas")
    palavras_unicas = len(set(palavras_filtradas))
    diversidade_lexical = palavras_unicas / total_palavras_filtradas
    
    # Etapa 4: Análise de repetição
    tracker.next_step("Analisando repetição de palavras", f"{palavras_unicas:,} palavras únicas ({diversidade_lexical:.3f})")
    contagem = Counter(palavras_filtradas)
    palavra_mais_comum, freq_mais_comum = contagem.most_common(1)[0]
    repeticao_excessiva = freq_mais_comum / total_palavras_filtradas
    
    # Etapa 5: Variação de sentenças
    tracker.next_step("Analisando variação de sentenças", f"'{palavra_mais_comum}': {freq_mais_comum}x ({repeticao_excessiva:.3f})")
    sentencas = [s.strip() for s in re.split(r'[.!?]+', texto) if s.strip()]
    total_sentencas = len(sentencas)
    
    if len(sentencas) > 1:
        comprimentos = [len(s.split()) for s in sentencas]
        variacao_sentencas = statistics.stdev(comprimentos) / statistics.mean(comprimentos) if statistics.mean(comprimentos) > 0 else 0
    else:
        variacao_sentencas = 0
    
    # Etapa 6: Repetição estrutural
    tracker.next_step("Analisando estrutura", f"{total_sentencas} sentenças (variação: {variacao_sentencas:.3f})")
    palavras_iniciais = []
    for sentenca in sentencas:
        palavras_sent = sentenca.split()
        if palavras_sent:
            palavras_iniciais.append(palavras_sent[0].lower())
    
    if palavras_iniciais:
        palavras_iniciais_unicas = len(set(palavras_iniciais))
        repeticao_estrutural = 1 - (palavras_iniciais_unicas / len(palavras_iniciais))
    else:
        repeticao_estrutural = 0
    
    # Etapa 7: Complexidade lexical
    tracker.next_step("Calculando complexidade lexical", f"Repetição estrutural: {repeticao_estrutural:.3f}")
    palavras_complexas = [p for p in palavras_filtradas if len(p) > 6]
    complexidade_lexical = len(palavras_complexas) / total_palavras_filtradas
    
    # Etapa 8: Uniformidade verbal
    tracker.next_step("Analisando uniformidade verbal", f"{len(palavras_complexas)} palavras complexas ({complexidade_lexical:.3f})")
    verbos_presente = ['é', 'são', 'está', 'estão', 'tem', 'têm', 'pode', 'podem', 'deve', 'devem']
    verbos_passado = ['foi', 'foram', 'estava', 'estavam', 'tinha', 'tinham', 'podia', 'podiam']
    verbos_futuro = ['será', 'serão', 'estará', 'estarão', 'terá', 'terão', 'poderá', 'poderão']
    
    freq_presente = sum(1 for palavra in palavras_completas if palavra in verbos_presente)
    freq_passado = sum(1 for palavra in palavras_completas if palavra in verbos_passado)
    freq_futuro = sum(1 for palavra in palavras_completas if palavra in verbos_futuro)
    
    total_verbos = freq_presente + freq_passado + freq_futuro
    if total_verbos > 0:
        uniformidade_verbal = max(freq_presente, freq_passado, freq_futuro) / total_verbos
    else:
        uniformidade_verbal = 0
    
    # Etapa 9: Cálculo final
    tracker.next_step("Calculando pontuação final", f"Verbos: P:{freq_presente} Pa:{freq_passado} F:{freq_futuro} (unif: {uniformidade_verbal:.3f})")
    
    # Análise de palavras típicas de IA
    palavras_ia_comum = ['sistema', 'processo', 'desenvolvimento', 'implementação', 'otimização',
                         'eficiência', 'benefício', 'vantagem', 'solução', 'abordagem', 'metodologia']
    palavras_ia_encontradas = [palavra for palavra in palavras_filtradas if palavra in palavras_ia_comum]
    freq_palavras_ia = len(palavras_ia_encontradas) / total_palavras_filtradas
    
    # Cálculo da pontuação de IA
    indicadores_ia = {
        'baixa_diversidade': (1 - diversidade_lexical) * 25,
        'repeticao_excessiva': repeticao_excessiva * 20,
        'baixa_variacao_sentencas': (1 - min(variacao_sentencas, 1)) * 15,
        'repeticao_estrutural': repeticao_estrutural * 15,
        'freq_palavras_ia': freq_palavras_ia * 100 * 15,
        'uniformidade_verbal': uniformidade_verbal * 10
    }
    
    # Log detalhado de cada indicador
    log_step("📊 INDICADORES DETALHADOS:", "")
    for indicador, pontos in indicadores_ia.items():
        log_step(f"   🎯 {indicador.replace('_', ' ').title()}", f"{pontos:.2f} pontos")
    
    probabilidade_ia = min(sum(indicadores_ia.values()), 100)
    
    # Determinação da confiança
    if probabilidade_ia >= 70:
        confianca = "Alta"
        mensagem = "🔴 ALTO RISCO: Texto apresenta múltiplas características típicas de IA"
    elif probabilidade_ia >= 40:
        confianca = "Média"
        mensagem = "🟡 RISCO MODERADO: Texto apresenta algumas características que podem indicar geração por IA"
    elif probabilidade_ia >= 20:
        confianca = "Baixa"
        mensagem = "🟢 BAIXO RISCO: Texto apresenta poucas características de IA"
    else:
        confianca = "Muito Baixa"
        mensagem = "✅ MUITO PROVÁVEL HUMANO: Texto apresenta características naturais de escrita humana"
    
    tracker.complete(f"Probabilidade: {probabilidade_ia:.1f}% | Confiança: {confianca}")
    
    detalhes = {
        'diversidade_lexical': round(diversidade_lexical, 3),
        'palavra_mais_comum': palavra_mais_comum,
        'frequencia_mais_comum': freq_mais_comum,
        'variacao_sentencas': round(variacao_sentencas, 3),
        'repeticao_estrutural': round(repeticao_estrutural * 100, 2),
        'complexidade_lexical': round(complexidade_lexical * 100, 2),
        'uniformidade_verbal': round(uniformidade_verbal * 100, 2),
        'indicadores_detalhados': {k: round(v, 2) for k, v in indicadores_ia.items()},
        'estatisticas_texto': {
            'total_palavras_completas': total_palavras_completas,
            'total_palavras_filtradas': total_palavras_filtradas,
            'total_sentencas': len(sentencas)
        }
    }
    
    return {
        "probabilidade_ia": round(probabilidade_ia, 1),
        "confianca": confianca,
        "detalhes": detalhes,
        "mensagem": mensagem,
        "diversidade_lexical": round(diversidade_lexical, 3),
        "palavra_mais_comum": palavra_mais_comum,
        "frequencia_mais_comum": freq_mais_comum
    }

def calcular_similaridade(texto1, texto2):
    """Calcula similaridade com progresso detalhado"""
    tracker = ProcessTracker("CÁLCULO DE SIMILARIDADE", 3)
    
    tracker.next_step("Validando textos", f"Texto 1: {len(texto1):,} chars | Texto 2: {len(texto2):,} chars")
    
    if not texto1.strip() or not texto2.strip():
        tracker.complete("ERRO: Texto vazio detectado")
        return 0.0
    
    try:
        tracker.next_step("Aplicando TF-IDF vectorizer", "Criando matriz de características")
        vetor = vectorizer.fit_transform([texto1, texto2])
        
        if vetor.shape[1] == 0:
            tracker.complete("ERRO: Nenhuma palavra válida")
            return 0.0
            
        tracker.next_step("Calculando cosine similarity", f"Matriz: {vetor.shape}")
        similaridade = float(cosine_similarity(vetor[0:1], vetor[1:2])[0][0])
        
        tracker.complete(f"Similaridade: {similaridade:.4f} ({similaridade*100:.2f}%)")
        return similaridade
        
    except Exception as e:
        tracker.complete(f"ERRO: {str(e)}")
        return 0.0

def encontrar_trechos_similares(texto1, texto2, top_n=3, limite_caracteres=400, bloco_palavras=50):
    """Busca de trechos similares ULTRA-OTIMIZADA com cálculo matricial em lote"""
    tracker = ProcessTracker("BUSCA DE TRECHOS SIMILARES", 5)
    
    def dividir_em_blocos(texto, tamanho):
        palavras = texto.split()
        # Otimização: criar blocos com sobreposição para melhor detecção
        blocos = []
        step = max(1, tamanho // 2)  # 50% de sobreposição
        for i in range(0, len(palavras), step):
            bloco = palavras[i:i + tamanho]
            if len(bloco) >= tamanho // 2:  # Aceitar blocos com pelo menos 50% do tamanho
                blocos.append(' '.join(bloco))
        return blocos

    tracker.next_step("Dividindo textos em blocos otimizados", f"Blocos de {bloco_palavras} palavras com 50% sobreposição")
    blocos1 = dividir_em_blocos(texto1, bloco_palavras)
    blocos2 = dividir_em_blocos(texto2, bloco_palavras)
    
    tracker.next_step("Validando blocos", f"Texto 1: {len(blocos1)} blocos | Texto 2: {len(blocos2)} blocos")
    
    if not blocos1 or not blocos2:
        tracker.complete("ERRO: Blocos insuficientes")
        return [(0.0, "Sem trecho suficiente", "Sem trecho suficiente")] * top_n

    # Otimização crítica: limitar número de blocos para textos muito grandes
    max_blocos = 200  # Limite para evitar explosão combinatória
    if len(blocos1) > max_blocos:
        # Selecionar blocos distribuídos uniformemente
        indices = [int(i * len(blocos1) / max_blocos) for i in range(max_blocos)]
        blocos1 = [blocos1[i] for i in indices]
        tracker.next_step("Limitando blocos texto 1", f"Reduzido para {len(blocos1)} blocos")
    
    if len(blocos2) > max_blocos:
        indices = [int(i * len(blocos2) / max_blocos) for i in range(max_blocos)]
        blocos2 = [blocos2[i] for i in indices]
        tracker.next_step("Limitando blocos texto 2", f"Reduzido para {len(blocos2)} blocos")

    total_comparacoes = len(blocos1) * len(blocos2)
    tracker.next_step("Preparando cálculo matricial", f"{total_comparacoes} comparações via operação matricial")
    
    try:
        # OTIMIZAÇÃO PRINCIPAL: Cálculo matricial em lote
        todos_blocos = blocos1 + blocos2
        
        tracker.next_step("Aplicando TF-IDF vetorizado", f"Processando {len(todos_blocos)} blocos em paralelo")
        
        # Criar vectorizer otimizado para esta operação
        vectorizer_local = TfidfVectorizer(
            stop_words=list(STOPWORDS),
            max_features=5000,  # Limitar features para velocidade
            ngram_range=(1, 2),  # Incluir bigramas para melhor precisão
            min_df=1,
            max_df=0.95
        )
        
        vetores = vectorizer_local.fit_transform(todos_blocos)
        
        if vetores.shape[1] == 0:
            tracker.complete("ERRO: Vetores vazios")
            return [(0.0, "Sem trecho suficiente", "Sem trecho suficiente")] * top_n
            
        n_blocos1 = len(blocos1)
        
        # MEGA OTIMIZAÇÃO: Cálculo matricial de TODAS as similaridades de uma vez
        log_step("🚀 EXECUTANDO cálculo matricial ultra-rápido", "Calculando todas as similaridades simultaneamente")
        
        # Separar vetores dos dois textos
        vetores1 = vetores[:n_blocos1]  # Primeiros n_blocos1 são do texto 1
        vetores2 = vetores[n_blocos1:]  # Restantes são do texto 2
        
        # Calcular matriz de similaridade completa de uma vez (MUITO mais rápido!)
        matriz_similaridade = cosine_similarity(vetores1, vetores2)
        
        tracker.next_step("Selecionando melhores trechos", f"Matriz {matriz_similaridade.shape} calculada instantaneamente")
        
        # Encontrar os top_n pares mais similares
        pares_similares = []
        
        # Achatar a matriz e encontrar os índices dos maiores valores
        indices_flat = matriz_similaridade.flatten().argsort()[-top_n*3:][::-1]  # Pegar 3x mais para filtrar depois
        
        for idx_flat in indices_flat:
            if len(pares_similares) >= top_n:
                break
                
            # Converter índice linear para coordenadas (i, j)
            i = idx_flat // matriz_similaridade.shape[1]
            j = idx_flat % matriz_similaridade.shape[1]
            
            sim = matriz_similaridade[i, j]
            
            # Filtrar similaridades muito baixas
            if sim < 0.1:  # Apenas similaridades >= 10%
                continue
                
            s1 = blocos1[i]
            s2 = blocos2[j]
            
            pares_similares.append((sim, s1, s2))
        
        # Preencher se necessário
        while len(pares_similares) < top_n:
            pares_similares.append((0.0, "Sem trecho suficiente", "Sem trecho suficiente"))

        # Truncar trechos para exibição
        resultado = []
        for sim, s1, s2 in pares_similares:
            s1_truncado = s1[:limite_caracteres] + '...' if len(s1) > limite_caracteres else s1
            s2_truncado = s2[:limite_caracteres] + '...' if len(s2) > limite_caracteres else s2
            resultado.append((sim, s1_truncado, s2_truncado))
        
        melhor_similaridade = resultado[0][0] if resultado else 0.0
        tracker.complete(f"✅ ULTRA-OTIMIZADO! Melhor similaridade: {melhor_similaridade:.3f}")
        
        return resultado
        
    except Exception as e:
        tracker.complete(f"ERRO: {str(e)}")
        return [(0.0, "Sem trecho suficiente", "Sem trecho suficiente")] * top_n

def gerar_pdf(resultado, trechos):
    """Geração otimizada de PDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Cabeçalho
    pdf.cell(200, 10, txt="Relatório de Comparação de Textos", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Índice de Correlação: {resultado * 100:.2f}%", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Trechos mais similares:", ln=True)

    # Conteúdo
    for i, (sim, s1, s2) in enumerate(trechos, 1):
        conteudo = f"Trecho {i} - Similaridade: {sim * 100:.2f}%\nTexto 1: {s1}\nTexto 2: {s2}\n"
        pdf.multi_cell(0, 10, txt=conteudo)
        pdf.ln(5)

    # Salvar em arquivo temporário
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    temp_file.close()
    return temp_file.name

def processar_arquivo(arquivo):
    """Função auxiliar para processar arquivos"""
    log_step("📁 INICIANDO processamento de arquivo", f"Nome: {arquivo.filename}")
    
    if arquivo.filename.endswith('.pdf'):
        log_step("📄 Tipo identificado: PDF")
        return extrair_texto_pdf(arquivo)
    else:
        log_step("📝 Tipo identificado: TXT")
        start_time = datetime.now()
        texto = arquivo.read().decode('utf-8')
        duracao = (datetime.now() - start_time).total_seconds()
        log_step("✅ TXT lido com sucesso", f"{len(texto)} caracteres em {duracao:.3f}s")
        return texto

def salvar_dados_temporarios(dados, prefix="data"):
    """Salva dados em arquivo temporário e retorna o ID"""
    log_step("💾 SALVANDO dados temporários", f"Prefix: {prefix}")
    start_time = datetime.now()
    
    arquivo_id = str(uuid.uuid4())
    caminho = TEMP_DIR / f"{prefix}_{arquivo_id}.pkl"
    
    try:
        with open(caminho, 'wb') as f:
            pickle.dump(dados, f)
        
        tamanho = caminho.stat().st_size
        duracao = (datetime.now() - start_time).total_seconds()
        log_step("✅ Dados salvos com sucesso", f"ID: {arquivo_id}, Tamanho: {tamanho} bytes, Tempo: {duracao:.3f}s")
        
        return arquivo_id
    except Exception as e:
        log_step("❌ ERRO ao salvar dados", f"Erro: {str(e)}")
        raise

def carregar_dados_temporarios(arquivo_id, prefix="data"):
    """Carrega dados do arquivo temporário"""
    log_step("📂 CARREGANDO dados temporários", f"ID: {arquivo_id}, Prefix: {prefix}")
    start_time = datetime.now()
    
    caminho = TEMP_DIR / f"{prefix}_{arquivo_id}.pkl"
    
    if not caminho.exists():
        log_step("⚠️ Arquivo não encontrado", f"Caminho: {caminho}")
        return None
        
    try:
        with open(caminho, 'rb') as f:
            dados = pickle.load(f)
        
        tamanho = caminho.stat().st_size
        duracao = (datetime.now() - start_time).total_seconds()
        log_step("✅ Dados carregados com sucesso", f"Tamanho: {tamanho} bytes, Tempo: {duracao:.3f}s")
        
        return dados
    except Exception as e:
        log_step("❌ ERRO ao carregar dados", f"Erro: {str(e)}")
        return None

def limpar_dados_temporarios(arquivo_id, prefix="data"):
    """Remove arquivo temporário"""
    log_step("🗑️ LIMPANDO arquivo temporário", f"ID: {arquivo_id}")
    
    caminho = TEMP_DIR / f"{prefix}_{arquivo_id}.pkl"
    try:
        if caminho.exists():
            caminho.unlink()
            log_step("✅ Arquivo removido", f"Caminho: {caminho}")
        else:
            log_step("⚠️ Arquivo já não existe", f"Caminho: {caminho}")
    except Exception as e:
        log_step("❌ ERRO ao remover arquivo", f"Erro: {str(e)}")

def limpar_arquivos_antigos():
    """Remove arquivos temporários com mais de 1 hora"""
    log_step("🧹 VERIFICANDO arquivos antigos...")
    import time
    now = time.time()
    arquivos_removidos = 0
    
    try:
        for arquivo in TEMP_DIR.glob("*.pkl"):
            try:
                if now - arquivo.stat().st_mtime > 3600:  # 1 hora
                    arquivo.unlink()
                    arquivos_removidos += 1
            except:
                pass
        
        if arquivos_removidos > 0:
            log_step("🗑️ Arquivos antigos removidos", f"{arquivos_removidos} arquivos limpos")
        else:
            log_step("✅ Nenhum arquivo antigo encontrado", "Sistema limpo")
            
    except Exception as e:
        log_step("❌ ERRO na limpeza automática", f"Erro: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    log_step("🌐 NOVA REQUISIÇÃO recebida", f"Método: {request.method}")
    
    # Limpar arquivos antigos periodicamente
    limpar_arquivos_antigos()
    
    if request.method == 'POST':
        modo = request.form.get('modo', 'comparar')
        log_step("🎯 Modo de operação", f"Modo: {modo}")
        
        if modo == 'comparar':
            try:
                log_step("📂 Obtendo arquivos do formulário...")
                arquivo1 = request.files['arquivo1']
                arquivo2 = request.files['arquivo2']
                log_step("📁 Arquivos recebidos", f"Arquivo 1: {arquivo1.filename}, Arquivo 2: {arquivo2.filename}")

                # Processamento dos textos
                log_step("🔄 INICIANDO processamento completo...")
                
                log_step("📄 Processando arquivo 1...")
                texto1 = limpar_texto(processar_arquivo(arquivo1))
                
                log_step("📄 Processando arquivo 2...")
                texto2 = limpar_texto(processar_arquivo(arquivo2))

                # Cálculos principais
                log_step("🧮 Executando cálculos principais...")
                
                resultado = calcular_similaridade(texto1, texto2)
                trechos = encontrar_trechos_similares(texto1, texto2)
                
                log_step("🤖 Analisando IA do arquivo 1...")
                analise_ia1 = analisar_texto_ia(texto1)
                
                log_step("🤖 Analisando IA do arquivo 2...")
                analise_ia2 = analisar_texto_ia(texto2)

                # Salvar dados em arquivos temporários
                log_step("💾 Salvando resultados em arquivo temporário...")
                dados_comparacao = {
                    'resultado': resultado,
                    'trechos': trechos,
                    'analise_ia1': analise_ia1,
                    'analise_ia2': analise_ia2,
                    'texto1': texto1,
                    'texto2': texto2,
                    'modo': 'comparar'
                }
                
                dados_id = salvar_dados_temporarios(dados_comparacao, "comparacao")
                log_step("💾 Dados salvos", f"ID: {dados_id}")

                # Salvar apenas o ID na sessão
                session.clear()
                session['dados_id'] = dados_id
                session['modo'] = 'comparar'
                
                log_step("✅ PROCESSAMENTO CONCLUÍDO", "Redirecionando para resultados")
                return redirect(url_for('index'))
                
            except Exception as e:
                log_step("❌ ERRO CRÍTICO no processamento", f"Erro: {str(e)}")
                return render_template('index.html', erro=f"Erro ao processar arquivos: {str(e)}")

    # GET request - recuperar dados dos arquivos temporários
    log_step("📖 Carregando página (GET)")
    dados_id = session.get('dados_id')
    modo = session.get('modo', 'comparar')
    
    if dados_id:
        log_step("🔍 ID encontrado na sessão", f"ID: {dados_id}, Modo: {modo}")
        
        if modo == 'comparar':
            dados = carregar_dados_temporarios(dados_id, "comparacao")
            if dados:
                log_step("✅ Dados de comparação carregados", "Exibindo resultados")
                return render_template(
                    'index.html',
                    resultado=dados.get('resultado'),
                    trechos=dados.get('trechos'),
                    analise_ia1=dados.get('analise_ia1', {}),
                    analise_ia2=dados.get('analise_ia2', {}),
                    modo='comparar'
                )
        elif modo == 'individual':
            dados = carregar_dados_temporarios(dados_id, "individual")
            if dados:
                log_step("✅ Dados individuais carregados", "Exibindo resultados")
                return render_template(
                    'index.html',
                    analise_individual=dados.get('analise_individual'),
                    modo='individual'
                )
    
    log_step("🏠 Exibindo página inicial", "Nenhum dado na sessão")
    return render_template('index.html')

@app.route('/analisar-individual', methods=['POST'])
def analisar_individual():
    log_step("🤖 NOVA ANÁLISE INDIVIDUAL iniciada")
    
    try:
        arquivo = request.files['arquivo']
        log_step("📁 Arquivo recebido", f"Nome: {arquivo.filename}")
        
        texto = limpar_texto(processar_arquivo(arquivo))
        
        # Análise individual de IA
        analise_individual = analisar_texto_ia(texto)
        
        # Salvar dados em arquivo temporário
        log_step("💾 Salvando análise individual...")
        dados_individual = {
            'analise_individual': analise_individual,
            'texto_individual': texto
        }
        
        dados_id = salvar_dados_temporarios(dados_individual, "individual")
        log_step("💾 Análise salva", f"ID: {dados_id}")
        
        # Salvar apenas o ID na sessão
        session.clear()
        session['dados_id'] = dados_id
        session['modo'] = 'individual'
        
        log_step("✅ ANÁLISE INDIVIDUAL CONCLUÍDA", "Redirecionando para resultados")
        return redirect(url_for('index'))
        
    except Exception as e:
        log_step("❌ ERRO na análise individual", f"Erro: {str(e)}")
        session['erro'] = f"Erro ao processar arquivo: {str(e)}"
        return redirect(url_for('index'))

def gerar_pdf_individual(analise):
    """Geração de PDF para análise individual"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Cabeçalho
    pdf.cell(200, 10, txt="Relatório de Análise Individual de IA", ln=True, align='C')
    pdf.ln(10)
    
    # Resultado principal
    pdf.cell(200, 10, txt=f"Probabilidade de IA: {analise['probabilidade_ia']}%", ln=True)
    pdf.cell(200, 10, txt=f"Confiança: {analise['confianca']}", ln=True)
    pdf.ln(5)
    
    # Mensagem
    pdf.multi_cell(0, 10, txt=f"Conclusão: {analise['mensagem']}")
    pdf.ln(10)
    
    # Métricas detalhadas
    if 'detalhes' in analise and analise['detalhes']:
        pdf.cell(200, 10, txt="Métricas Detalhadas:", ln=True)
        pdf.ln(5)
        
        detalhes = analise['detalhes']
        conteudo_detalhes = f"""
Diversidade Lexical: {detalhes.get('diversidade_lexical', 'N/A')}
Variação de Sentenças: {detalhes.get('variacao_sentencas', 'N/A')}
Repetição Estrutural: {detalhes.get('repeticao_estrutural', 'N/A')}%
Complexidade Lexical: {detalhes.get('complexidade_lexical', 'N/A')}
Uniformidade Verbal: {detalhes.get('uniformidade_verbal', 'N/A')}
Palavra mais comum: "{detalhes.get('palavra_mais_comum', 'N/A')}" ({detalhes.get('frequencia_mais_comum', 'N/A')} vezes)
        """
        
        pdf.multi_cell(0, 8, txt=conteudo_detalhes.strip())
        
        # Indicadores detalhados
        if 'indicadores_detalhados' in detalhes:
            pdf.ln(10)
            pdf.cell(200, 10, txt="Pontuação por Indicador:", ln=True)
            pdf.ln(5)
            
            for indicador, pontos in detalhes['indicadores_detalhados'].items():
                nome_indicador = indicador.replace('_', ' ').title()
                pdf.cell(0, 8, txt=f"• {nome_indicador}: {pontos:.2f} pontos", ln=True)

    # Salvar em arquivo temporário
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    temp_file.close()
    return temp_file.name

@app.route('/baixar-relatorio')
def baixar_relatorio():
    """Gerar e baixar relatório otimizado"""
    try:
        dados_id = session.get('dados_id')
        if not dados_id:
            return redirect(url_for('index'))
            
        dados = carregar_dados_temporarios(dados_id, "comparacao")
        if not dados:
            return redirect(url_for('index'))
            
        resultado = dados.get('resultado', 0)
        trechos = dados.get('trechos', [])
        
        caminho_pdf = gerar_pdf(resultado, trechos)
        return send_file(caminho_pdf, as_attachment=True, 
                        download_name='relatorio_comparacao.pdf', 
                        mimetype='application/pdf')
    except Exception as e:
        return redirect(url_for('index'))

@app.route('/baixar-relatorio-individual')
def baixar_relatorio_individual():
    """Gerar e baixar relatório individual"""
    try:
        dados_id = session.get('dados_id')
        if not dados_id:
            return redirect(url_for('index'))
            
        dados = carregar_dados_temporarios(dados_id, "individual")
        if not dados:
            return redirect(url_for('index'))
            
        analise_individual = dados.get('analise_individual')
        if not analise_individual:
            return redirect(url_for('index'))
            
        caminho_pdf = gerar_pdf_individual(analise_individual)
        return send_file(caminho_pdf, as_attachment=True, 
                        download_name='relatorio_analise_ia.pdf', 
                        mimetype='application/pdf')
    except Exception as e:
        return redirect(url_for('index'))

# Limpeza automática ao fechar a aplicação
import atexit

def cleanup():
    """Limpa todos os arquivos temporários ao fechar"""
    try:
        for arquivo in TEMP_DIR.glob("*.pkl"):
            arquivo.unlink()
    except:
        pass

atexit.register(cleanup)

if __name__ == '__main__':
    app.run(debug=True)
