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
    tracker = ProcessTracker("EXTRAÇÃO PDF", 5)
    
    tracker.next_step("Carregando PDF", f"Arquivo: {arquivo.filename}")
    leitor = PdfReader(arquivo)
    total_paginas = len(leitor.pages)
    
    tracker.next_step("Analisando estrutura", f"{total_paginas} páginas encontradas")
    
    tracker.next_step("Extraindo texto das páginas", "Iniciando extração sequencial")
    texto_completo = ""
    
    for i, pagina in enumerate(leitor.pages, 1):
        texto_pagina = pagina.extract_text() or ""
        
        # Adicionar espaço entre páginas se necessário
        if texto_completo and texto_pagina:
            # Se a página anterior não termina com espaço e a atual não começa com espaço
            if not texto_completo.endswith(' ') and not texto_pagina.startswith(' '):
                texto_completo += " "
        
        texto_completo += texto_pagina
        
        # Log detalhado do progresso
        tracker.substep(3, f"Página {i}", i, total_paginas, f"{len(texto_pagina)} caracteres extraídos")
    
    tracker.next_step("Corrigindo problemas comuns de extração PDF", "Aplicando correções específicas")
    
    # Correções específicas para problemas comuns do PyPDF2
    texto_corrigido = texto_completo
    
    # 1. Corrigir palavras que ficaram grudadas sem espaço
    # Detectar transições de letra minúscula para maiúscula (nova palavra/frase)
    import re
    texto_corrigido = re.sub(r'([a-záàâãéêíóôõúç])([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])', r'\1 \2', texto_corrigido)
    
    # 2. Corrigir números grudados com palavras
    texto_corrigido = re.sub(r'([0-9])([a-zA-ZáàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ])', r'\1 \2', texto_corrigido)
    texto_corrigido = re.sub(r'([a-zA-ZáàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ])([0-9])', r'\1 \2', texto_corrigido)
    
    # 3. Corrigir pontuação grudada
    texto_corrigido = re.sub(r'([.!?;:])([a-zA-ZáàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ])', r'\1 \2', texto_corrigido)
    
    # 4. NOVO: Corrigir espaços indevidos dentro de palavras
    # Remove espaços entre letras que claramente formam uma palavra
    # Ex: "p a l a v r a" -> "palavra"
    texto_corrigido = re.sub(r'\b([a-záàâãéêíóôõúç])\s+([a-záàâãéêíóôõúç])\s+([a-záàâãéêíóôõúç])', r'\1\2\3', texto_corrigido)
    texto_corrigido = re.sub(r'\b([a-záàâãéêíóôõúç])\s+([a-záàâãéêíóôõúç])\b', r'\1\2', texto_corrigido)
    
    # 5. NOVO: Corrigir palavras quebradas por espaços (padrão mais agressivo)
    # Identifica sequências como "ex em plo" que deveriam ser "exemplo"
    def corrigir_palavras_espacadas(match):
        palavra_quebrada = match.group(0)
        # Remove espaços e reconstitui a palavra
        palavra_corrigida = re.sub(r'\s+', '', palavra_quebrada)
        # Verifica se a palavra reconstituída faz sentido (tem pelo menos 4 caracteres)
        if len(palavra_corrigida) >= 4:
            return palavra_corrigida
        return palavra_quebrada
    
    # Aplicar correção para palavras com 2-3 letras separadas por espaços
    texto_corrigido = re.sub(r'\b[a-záàâãéêíóôõúç]{1,2}\s+[a-záàâãéêíóôõúç]{1,2}\s+[a-záàâãéêíóôõúç]{1,2}\b', corrigir_palavras_espacadas, texto_corrigido)
    
    # 6. NOVO: Corrigir espaços antes de sufixos comuns
    # Ex: "desenvolv imento" -> "desenvolvimento"
    sufixos = ['ção', 'mento', 'agem', 'ando', 'endo', 'indo', 'mente', 'dade', 'ente', 'ante']
    for sufixo in sufixos:
        pattern = r'([a-záàâãéêíóôõúç]+)\s+(' + sufixo + r')\b'
        texto_corrigido = re.sub(pattern, r'\1\2', texto_corrigido)
    
    # 7. NOVO: Corrigir espaços após prefixos comuns
    # Ex: "des envolvimento" -> "desenvolvimento"
    prefixos = ['des', 'pre', 'sub', 'super', 'anti', 'contra', 'inter', 'multi', 'semi', 're']
    for prefixo in prefixos:
        pattern = r'\b(' + prefixo + r')\s+([a-záàâãéêíóôõúç]+)'
        texto_corrigido = re.sub(pattern, r'\1\2', texto_corrigido)
    
    # 8. Corrigir espaços antes de pontuação (comum em PDFs mal formatados)
    texto_corrigido = re.sub(r'\s+([.!?;:,])', r'\1', texto_corrigido)
    
    # 9. Normalizar quebras de linha múltiplas
    texto_corrigido = re.sub(r'\n+', ' ', texto_corrigido)
    
    # 10. Corrigir palavras quebradas por hífens (comum quando texto é quebrado em linhas)
    texto_corrigido = re.sub(r'([a-záàâãéêíóôõúç])-\s+([a-záàâãéêíóôõúç])', r'\1\2', texto_corrigido)
    
    # 11. NOVO: Corrigir espaços duplos em sequências específicas
    # Para casos como "c o m o" -> "como", "t e x t o" -> "texto"
    def corrigir_letras_isoladas(texto):
        # Identifica padrões de letras isoladas que formam palavras conhecidas
        palavras_comuns = ['como', 'texto', 'para', 'este', 'esta', 'mais', 'muito', 'pode', 'deve', 'sobre', 'quando', 'depois', 'antes', 'exemplo', 'análise', 'sistema', 'processo', 'projeto', 'trabalho', 'estudo', 'pesquisa', 'resultado', 'método', 'técnica', 'aplicação']
        for palavra in palavras_comuns:
            # Cria padrão para palavra espacada: "c o m o"
            pattern = r'\b' + r'\s+'.join(list(palavra)) + r'\b'
            texto = re.sub(pattern, palavra, texto, flags=re.IGNORECASE)
        return texto
    
    texto_corrigido = corrigir_letras_isoladas(texto_corrigido)
    
    # 12. NOVO: Detectar e corrigir padrões de palavras com espaços internos
    def detectar_palavras_espacadas(texto):
        # Padrão mais inteligente: identifica sequências de 1-2 letras separadas por espaços
        # que provavelmente formam uma palavra
        pattern = r'\b(?:[a-záàâãéêíóôõúç]{1,2}\s+){2,}[a-záàâãéêíóôõúç]{1,2}\b'
        
        def corrigir_match(match):
            sequencia = match.group(0)
            # Remove espaços para formar palavra
            palavra_possivel = re.sub(r'\s+', '', sequencia)
            
            # Se a palavra resultante tem tamanho razoável, provavelmente está correta
            if 4 <= len(palavra_possivel) <= 15:
                return palavra_possivel
            return sequencia
        
        return re.sub(pattern, corrigir_match, texto)
    
    texto_corrigido = detectar_palavras_espacadas(texto_corrigido)
    
    # 13. Normalizar espaços múltiplos (final)
    texto_corrigido = re.sub(r'\s+', ' ', texto_corrigido)
    
    caracteres_originais = len(texto_completo)
    caracteres_corrigidos = len(texto_corrigido)
    
    tracker.next_step("Finalizando extração", f"Original: {caracteres_originais:,} → Corrigido: {caracteres_corrigidos:,} caracteres")
    
    tracker.complete(f"PDF processado: {total_paginas} páginas, {caracteres_corrigidos:,} caracteres finais")
    return texto_corrigido.strip()

def limpar_texto(texto):
    """Limpa texto com progresso detalhado"""
    tracker = ProcessTracker("LIMPEZA DE TEXTO", 4)
    
    original_len = len(texto)
    tracker.next_step("Aplicando regex de espaços", f"Texto original: {original_len:,} caracteres")
    
    # Remover quebras de linha e espaços extras
    texto_limpo = TEXTO_REGEX.sub(' ', texto)
    after_spaces = len(texto_limpo)
    spaces_removed = original_len - after_spaces
    
    tracker.next_step("Removendo caracteres especiais", f"Após espaços: {after_spaces:,} caracteres (-{spaces_removed:,})")
    
    # Remover caracteres especiais
    texto_limpo = CHAR_REGEX.sub('', texto_limpo)
    after_chars = len(texto_limpo)
    chars_removed = after_spaces - after_chars
    
    tracker.next_step("Normalizando espaços múltiplos", f"Após remoção de caracteres: {after_chars:,} caracteres")
    
    # CORREÇÃO: Normalizar espaços múltiplos que ficaram após remoção de caracteres especiais
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo)
    texto_final = texto_limpo.strip()
    final_len = len(texto_final)
    spaces_normalized = after_chars - final_len
    total_removed = original_len - final_len
    
    tracker.next_step("Finalizando limpeza", f"Espaços normalizados: {spaces_normalized:,} | Total removido: {total_removed:,}")
    
    tracker.complete(f"Limpeza finalizada: {final_len:,} caracteres finais (-{total_removed:,} total)")
    return texto_final

def limpar_texto_para_analise(texto):
    """Limpa texto especificamente para análise (remove pontuação e caracteres especiais)"""
    # Remover quebras de linha e espaços extras
    texto_limpo = TEXTO_REGEX.sub(' ', texto)
    # Remover caracteres especiais mas manter espaços
    texto_limpo = CHAR_REGEX.sub('', texto_limpo)
    # CORREÇÃO: Normalizar espaços múltiplos que ficaram após remoção
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo)
    return texto_limpo.strip()

def limpar_texto_basico(texto):
    """Limpeza básica mantendo pontuação (apenas normaliza espaços)"""
    # Apenas normalizar espaços, mantendo pontuação
    texto_limpo = TEXTO_REGEX.sub(' ', texto)
    # CORREÇÃO: Garantir que não há espaços múltiplos
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo)
    return texto_limpo.strip()

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

def encontrar_trechos_similares(texto1, texto2, top_n=3, limite_caracteres=400, bloco_palavras=100):
    """Busca de trechos similares com texto original para exibição e texto filtrado para análise"""
    tracker = ProcessTracker("BUSCA DE TRECHOS SIMILARES", 6)
    
    def dividir_em_blocos_original(texto, tamanho):
        """Divide texto original mantendo pontuação e formatação para exibição"""
        # Aplicar apenas limpeza básica (normalizar espaços, manter pontuação)
        texto_limpo = limpar_texto_basico(texto)
        palavras = texto_limpo.split()
        blocos = []
        step = max(1, tamanho // 3)  # 33% de sobreposição para melhor cobertura
        for i in range(0, len(palavras), step):
            bloco = palavras[i:i + tamanho]
            if len(bloco) >= tamanho // 3:  # Aceitar blocos menores no final
                blocos.append(' '.join(bloco))
        return blocos
    
    def dividir_em_blocos_filtrados(texto, tamanho):
        """Divide texto filtrado (sem stopwords e caracteres especiais) para análise de similaridade"""
        # Aplicar limpeza completa para análise
        texto_limpo = limpar_texto_para_analise(texto)
        palavras = texto_limpo.split()
        # Filtrar stopwords e palavras muito pequenas
        palavras_filtradas = [p for p in palavras if p.lower() not in STOPWORDS and len(p) > 1]
        
        blocos = []
        step = max(1, tamanho // 3)  # 33% de sobreposição para melhor cobertura
        for i in range(0, len(palavras_filtradas), step):
            bloco = palavras_filtradas[i:i + tamanho]
            if len(bloco) >= tamanho // 3:  # Aceitar blocos menores no final
                blocos.append(' '.join(bloco))
        return blocos
    
    def analisar_similaridade_detalhada(bloco1_filtrado, bloco2_filtrado, bloco1_original, bloco2_original):
        """Analisa e explica por que dois blocos são similares"""
        # Converter para sets de palavras (sem duplicatas)
        palavras1 = set(bloco1_filtrado.lower().split())
        palavras2 = set(bloco2_filtrado.lower().split())
        
        # Encontrar palavras em comum
        palavras_comuns = palavras1.intersection(palavras2)
        
        # Calcular métricas de similaridade
        total_palavras_unicas = len(palavras1.union(palavras2))
        jaccard_similarity = len(palavras_comuns) / total_palavras_unicas if total_palavras_unicas > 0 else 0
        
        # Identificar tipos de palavras em comum
        palavras_importantes = []
        palavras_conceitos = []
        palavras_numericas = []
        palavras_tecnicas = []
        
        for palavra in palavras_comuns:
            if len(palavra) >= 8:  # Palavras muito longas (mais específicas)
                palavras_importantes.append(palavra)
            elif len(palavra) >= 6:  # Palavras longas
                palavras_tecnicas.append(palavra)
            if any(char.isdigit() for char in palavra):  # Contém números
                palavras_numericas.append(palavra)
            # Palavras conceituais (substantivos técnicos comuns)
            if palavra in ['sistema', 'processo', 'método', 'análise', 'desenvolvimento', 'projeto', 'resultado', 'dados', 'informação', 'técnica', 'aplicação', 'trabalho', 'estudo', 'pesquisa', 'modelo', 'estrutura', 'função', 'operação', 'gestão', 'controle', 'qualidade', 'eficiência', 'performance', 'implementação', 'solução', 'problema', 'questão', 'objetivo', 'meta', 'estratégia', 'planejamento', 'conhecimento', 'aprendizagem', 'educação', 'ensino', 'metodologia', 'abordagem', 'tecnologia', 'inovação', 'recurso', 'ferramenta', 'plataforma', 'ambiente']:
                palavras_conceitos.append(palavra)
        
        # Detectar padrões sintáticos similares
        # Contar palavras funcionais importantes que dão estrutura
        palavras_estruturais_comuns = palavras_comuns.intersection({'através', 'mediante', 'conforme', 'segundo', 'durante', 'enquanto', 'portanto', 'entretanto', 'contudo', 'assim', 'então', 'logo', 'pois', 'porque', 'quando', 'onde', 'como', 'quanto', 'qual', 'quais', 'sendo', 'tendo', 'fazendo', 'utilizando', 'empregando', 'considerando', 'verificando'})
        
        # Detectar sequências de palavras (bigramas comuns)
        palavras1_lista = bloco1_filtrado.lower().split()
        palavras2_lista = bloco2_filtrado.lower().split()
        
        bigramas1 = set([f"{palavras1_lista[i]} {palavras1_lista[i+1]}" for i in range(len(palavras1_lista)-1)])
        bigramas2 = set([f"{palavras2_lista[i]} {palavras2_lista[i+1]}" for i in range(len(palavras2_lista)-1)])
        bigramas_comuns = bigramas1.intersection(bigramas2)
        
        # Gerar explicação
        explicacao_partes = []
        
        # Priorizar explicações mais específicas
        if palavras_importantes:
            top_importantes = sorted(palavras_importantes, key=len, reverse=True)[:2]
            explicacao_partes.append(f"Termos específicos: {', '.join(top_importantes)}")
            
        if palavras_conceitos:
            explicacao_partes.append(f"Conceitos técnicos: {', '.join(palavras_conceitos[:3])}")
            
        if bigramas_comuns and len(bigramas_comuns) >= 2:
            top_bigramas = list(bigramas_comuns)[:2]
            explicacao_partes.append(f"Expressões similares: {', '.join(top_bigramas)}")
            
        if palavras_numericas:
            explicacao_partes.append(f"Elementos numéricos: {', '.join(palavras_numericas[:3])}")
            
        if palavras_tecnicas and not palavras_importantes:  # Só mostrar se não tiver palavras mais importantes
            top_tecnicas = sorted(palavras_tecnicas, key=len, reverse=True)[:3]
            explicacao_partes.append(f"Termos técnicos: {', '.join(top_tecnicas)}")
            
        if palavras_estruturais_comuns:
            explicacao_partes.append(f"Estrutura textual similar")
            
        # Análise de densidade de palavras comuns
        densidade1 = len(palavras_comuns) / len(palavras1) if len(palavras1) > 0 else 0
        densidade2 = len(palavras_comuns) / len(palavras2) if len(palavras2) > 0 else 0
        densidade_media = (densidade1 + densidade2) / 2
        
        # Mostrar quantidade de palavras compartilhadas se significativa
        if len(palavras_comuns) >= 5:
            explicacao_partes.append(f"Compartilham {len(palavras_comuns)} palavras-chave")
        
        if densidade_media > 0.4:
            explicacao_partes.append("Alta densidade de termos comuns")
        elif densidade_media > 0.25:
            explicacao_partes.append("Densidade moderada de termos comuns")
            
        # Se não encontrou motivos específicos, dar explicação mais detalhada
        if not explicacao_partes:
            if len(palavras_comuns) >= 3:
                top_comuns = sorted(list(palavras_comuns), key=len, reverse=True)[:4]
                explicacao_partes.append(f"Palavras em comum: {', '.join(top_comuns)}")
            elif len(palavras_comuns) > 0:
                explicacao_partes.append(f"Algumas palavras compartilhadas: {', '.join(list(palavras_comuns)[:3])}")
            else:
                explicacao_partes.append("Similaridade semântica detectada pelo algoritmo TF-IDF")
        
        explicacao = " • ".join(explicacao_partes)
        
        return {
            'explicacao': explicacao,
            'palavras_comuns': list(palavras_comuns),
            'densidade_similaridade': densidade_media,
            'jaccard_similarity': jaccard_similarity,
            'bigramas_comuns': list(bigramas_comuns)
        }

    tracker.next_step("Preparando textos originais para exibição", f"Mantendo pontuação e formatação completa")
    # Textos originais (com pontuação) para exibição
    blocos1_original = dividir_em_blocos_original(texto1, bloco_palavras)
    blocos2_original = dividir_em_blocos_original(texto2, bloco_palavras)
    
    tracker.next_step("Preparando textos filtrados para análise", f"Removendo stopwords e caracteres especiais")
    # Textos filtrados para análise de similaridade
    blocos1_filtrados = dividir_em_blocos_filtrados(texto1, bloco_palavras)
    blocos2_filtrados = dividir_em_blocos_filtrados(texto2, bloco_palavras)
    
    tracker.next_step("Validando blocos", f"Original: T1={len(blocos1_original)}, T2={len(blocos2_original)} | Filtrados: T1={len(blocos1_filtrados)}, T2={len(blocos2_filtrados)}")
    
    if not blocos1_filtrados or not blocos2_filtrados:
        tracker.complete("ERRO: Blocos filtrados insuficientes para análise")
        return [(0.0, "Sem trecho suficiente", "Sem trecho suficiente", "Nenhuma similaridade encontrada", "Sem trecho suficiente", "Sem trecho suficiente", False, False)] * top_n

    # Otimização crítica: limitar número de blocos para textos muito grandes
    max_blocos = 150  # Reduzido um pouco para acomodar blocos maiores
    
    if len(blocos1_filtrados) > max_blocos:
        # Selecionar blocos distribuídos uniformemente, mantendo correspondência entre original e filtrado
        indices = [int(i * len(blocos1_filtrados) / max_blocos) for i in range(max_blocos)]
        blocos1_filtrados = [blocos1_filtrados[i] for i in indices]
        # Garantir que temos blocos originais correspondentes
        blocos1_original = [blocos1_original[min(i, len(blocos1_original)-1)] for i in indices]
        tracker.next_step("Limitando blocos texto 1", f"Reduzido para {len(blocos1_filtrados)} blocos")
    
    if len(blocos2_filtrados) > max_blocos:
        indices = [int(i * len(blocos2_filtrados) / max_blocos) for i in range(max_blocos)]
        blocos2_filtrados = [blocos2_filtrados[i] for i in indices]
        blocos2_original = [blocos2_original[min(i, len(blocos2_original)-1)] for i in indices]
        tracker.next_step("Limitando blocos texto 2", f"Reduzido para {len(blocos2_filtrados)} blocos")

    total_comparacoes = len(blocos1_filtrados) * len(blocos2_filtrados)
    tracker.next_step("Preparando cálculo matricial", f"{total_comparacoes} comparações via operação matricial")
    
    try:
        # OTIMIZAÇÃO PRINCIPAL: Cálculo matricial em lote usando blocos FILTRADOS
        todos_blocos_filtrados = blocos1_filtrados + blocos2_filtrados
        
        tracker.next_step("Aplicando TF-IDF vetorizado", f"Processando {len(todos_blocos_filtrados)} blocos filtrados em paralelo")
        
        # Criar vectorizer otimizado para esta operação
        vectorizer_local = TfidfVectorizer(
            stop_words=list(STOPWORDS),
            max_features=5000,  # Limitar features para velocidade
            ngram_range=(1, 2),  # Incluir bigramas para melhor precisão
            min_df=1,
            max_df=0.95
        )
        
        vetores = vectorizer_local.fit_transform(todos_blocos_filtrados)
        
        if vetores.shape[1] == 0:
            tracker.complete("ERRO: Vetores vazios")
            return [(0.0, "Sem trecho suficiente", "Sem trecho suficiente", "Nenhuma similaridade encontrada", "Sem trecho suficiente", "Sem trecho suficiente", False, False)] * top_n
            
        n_blocos1 = len(blocos1_filtrados)
        
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
            if sim < 0.15:  # Aumentado para 15% com blocos maiores
                continue
                
            # IMPORTANTE: Usar blocos ORIGINAIS para exibição (não os filtrados)
            s1_original = blocos1_original[min(i, len(blocos1_original)-1)]
            s2_original = blocos2_original[min(j, len(blocos2_original)-1)]
            
            # NOVO: Analisar e explicar a similaridade
            s1_filtrado = blocos1_filtrados[i] if i < len(blocos1_filtrados) else ""
            s2_filtrado = blocos2_filtrados[j] if j < len(blocos2_filtrados) else ""
            
            analise_detalhada = analisar_similaridade_detalhada(s1_filtrado, s2_filtrado, s1_original, s2_original)
            
            pares_similares.append((sim, s1_original, s2_original, analise_detalhada))
        
        # Preencher se necessário
        while len(pares_similares) < top_n:
            pares_similares.append((0.0, "Sem trecho suficiente", "Sem trecho suficiente", {'explicacao': 'Nenhuma similaridade encontrada'}))

        # Truncar trechos para exibição, preservando palavras completas e pontuação
        resultado = []
        for item in pares_similares:
            if len(item) == 4:  # Nova estrutura com análise detalhada
                sim, s1_original, s2_original, analise_detalhada = item
            else:  # Estrutura antiga (compatibilidade)
                sim, s1_original, s2_original = item
                analise_detalhada = {'explicacao': 'Similaridade detectada pelo algoritmo'}
            
            # Função melhorada para truncar preservando integridade do texto
            def truncar_texto_inteligente(texto, max_chars=limite_caracteres):
                if len(texto) <= max_chars:
                    return texto
                
                # Tentar cortar em pontuação final
                pontos_finais = ['.', '!', '?']
                for ponto in pontos_finais:
                    pos = texto.rfind(ponto, 0, max_chars)
                    if pos > max_chars * 0.7:  # Pelo menos 70% do tamanho desejado
                        return texto[:pos+1]
                
                # Tentar cortar em pontuação intermediária
                pontos_intermedios = [';', ',']
                for ponto in pontos_intermedios:
                    pos = texto.rfind(ponto, 0, max_chars)
                    if pos > max_chars * 0.8:  # Pelo menos 80% do tamanho desejado
                        return texto[:pos+1] + '...'
                
                # Se não encontrou pontuação, cortar no último espaço
                pos = texto.rfind(' ', 0, max_chars)
                if pos > max_chars * 0.8:  # Pelo menos 80% do tamanho desejado
                    return texto[:pos] + '...'
                
                # Em último caso, cortar direto mas sem quebrar palavras
                return texto[:max_chars-3] + '...'
            
            s1_truncado = truncar_texto_inteligente(s1_original)
            s2_truncado = truncar_texto_inteligente(s2_original)
            
            # Verificar se o texto foi realmente truncado
            s1_foi_truncado = len(s1_truncado) < len(s1_original)
            s2_foi_truncado = len(s2_truncado) < len(s2_original)
            
            # Nova estrutura: (similaridade, texto1_truncado, texto2_truncado, explicacao, texto1_completo, texto2_completo, foi_truncado1, foi_truncado2)
            resultado.append((sim, s1_truncado, s2_truncado, analise_detalhada['explicacao'], s1_original, s2_original, s1_foi_truncado, s2_foi_truncado))
        
        melhor_similaridade = resultado[0][0] if resultado else 0.0
        tracker.complete(f"✅ ULTRA-OTIMIZADO! Melhor similaridade: {melhor_similaridade:.3f}")
        
        return resultado
        
    except Exception as e:
        tracker.complete(f"ERRO: {str(e)}")
        return [(0.0, "Sem trecho suficiente", "Sem trecho suficiente", "Erro na análise", "Sem trecho suficiente", "Sem trecho suficiente", False, False)] * top_n

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
    for i, item in enumerate(trechos, 1):
        if len(item) >= 6:  # Nova estrutura com textos completos
            sim = item[0]
            explicacao = item[3] if len(item) > 3 else "Similaridade detectada pelo algoritmo"
            # Usar textos completos para o PDF (índices 4 e 5)
            s1 = item[4] if len(item) > 4 else item[1]
            s2 = item[5] if len(item) > 5 else item[2]
        elif len(item) == 4:  # Estrutura com explicação
            sim, s1, s2, explicacao = item
        else:  # Estrutura antiga (compatibilidade)
            sim, s1, s2 = item
            explicacao = "Similaridade detectada pelo algoritmo"
            
        conteudo = f"Trecho {i} - Similaridade: {sim * 100:.2f}%\n"
        conteudo += f"Motivo: {explicacao}\n\n"
        conteudo += f"Texto 1: {s1}\n\n"
        conteudo += f"Texto 2: {s2}\n"
        
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
                texto1_bruto = processar_arquivo(arquivo1)
                texto1_limpo = limpar_texto(texto1_bruto)  # Para análise de IA e similaridade geral
                
                log_step("📄 Processando arquivo 2...")
                texto2_bruto = processar_arquivo(arquivo2)
                texto2_limpo = limpar_texto(texto2_bruto)  # Para análise de IA e similaridade geral

                # Cálculos principais
                log_step("🧮 Executando cálculos principais...")
                
                # Usar texto limpo para cálculo geral de similaridade
                resultado = calcular_similaridade(texto1_limpo, texto2_limpo)
                
                # Usar texto original (com pontuação) para encontrar trechos similares
                trechos = encontrar_trechos_similares(texto1_bruto, texto2_bruto)
                
                log_step("🤖 Analisando IA do arquivo 1...")
                analise_ia1 = analisar_texto_ia(texto1_limpo)
                
                log_step("🤖 Analisando IA do arquivo 2...")
                analise_ia2 = analisar_texto_ia(texto2_limpo)

                # Salvar dados em arquivos temporários
                log_step("💾 Salvando resultados em arquivo temporário...")
                dados_comparacao = {
                    'resultado': resultado,
                    'trechos': trechos,
                    'analise_ia1': analise_ia1,
                    'analise_ia2': analise_ia2,
                    'texto1': texto1_limpo,  # Para exibição das análises
                    'texto2': texto2_limpo,  # Para exibição das análises
                    'texto1_bruto': texto1_bruto,  # Para debug
                    'texto2_bruto': texto2_bruto,  # Para debug
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

@app.route('/debug-texto')
def debug_texto():
    """Rota para visualizar o texto bruto extraído (debug)"""
    dados_id = session.get('dados_id')
    if not dados_id:
        return "Nenhum dado na sessão. Faça upload de arquivos primeiro."
    
    dados = carregar_dados_temporarios(dados_id, "comparacao")
    if not dados:
        return "Dados não encontrados."
    
    # Função para escapar HTML
    def escape_html(text):
        if not text:
            return 'Não disponível'
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    texto1_bruto = dados.get('texto1_bruto', '')
    texto2_bruto = dados.get('texto2_bruto', '')
    texto1_limpo = dados.get('texto1', '')
    texto2_limpo = dados.get('texto2', '')
    
    # Função para mostrar amostra do texto
    def amostra_texto(texto, limite=1500):
        if not texto:
            return 'Não disponível'
        texto_escapado = escape_html(texto[:limite])
        if len(texto) > limite:
            texto_escapado += '\n\n... (truncado)'
        return texto_escapado
    
    debug_info = f"""
    <html>
    <head>
        <title>Debug - Texto Extraído</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .texto-container {{ border: 1px solid #ccc; margin: 20px 0; padding: 15px; background: #f9f9f9; }}
            .texto-bruto {{ font-family: monospace; white-space: pre-wrap; max-height: 400px; overflow-y: auto; 
                           border: 1px solid #ddd; padding: 10px; background: white; }}
            .stats {{ background: #e6f3ff; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; 
                       border-radius: 5px; color: #856404; }}
            .comparison {{ display: flex; gap: 20px; }}
            .column {{ flex: 1; }}
            h3 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
        </style>
    </head>
    <body>
        <h1>🔍 Debug - Texto Extraído dos Arquivos</h1>
        <p><a href="/">← Voltar para a página principal</a></p>
        
        <div class="warning">
            <strong>⚠️ Modo Debug:</strong> Esta página mostra os textos extraídos para diagnóstico. 
            Verifique se as pontuações e formatações estão preservadas corretamente.
        </div>
        
        <div class="stats">
            <h3>📊 Estatísticas dos Textos:</h3>
            <div class="comparison">
                <div class="column">
                    <p><strong>📄 Arquivo 1:</strong></p>
                    <p>• Texto Bruto: {len(texto1_bruto):,} caracteres</p>
                    <p>• Texto Limpo: {len(texto1_limpo):,} caracteres</p>
                    <p>• Redução: {len(texto1_bruto) - len(texto1_limpo):,} caracteres</p>
                </div>
                <div class="column">
                    <p><strong>📄 Arquivo 2:</strong></p>
                    <p>• Texto Bruto: {len(texto2_bruto):,} caracteres</p>
                    <p>• Texto Limpo: {len(texto2_limpo):,} caracteres</p>
                    <p>• Redução: {len(texto2_bruto) - len(texto2_limpo):,} caracteres</p>
                </div>
            </div>
        </div>
        
        <div class="comparison">
            <div class="column">
                <div class="texto-container">
                    <h3>📄 Texto 1 - BRUTO (Original extraído)</h3>
                    <div class="texto-bruto">{amostra_texto(texto1_bruto)}</div>
                </div>
                
                <div class="texto-container">
                    <h3>🧹 Texto 1 - LIMPO (Após processamento)</h3>
                    <div class="texto-bruto">{amostra_texto(texto1_limpo)}</div>
                </div>
            </div>
            
            <div class="column">
                <div class="texto-container">
                    <h3>📄 Texto 2 - BRUTO (Original extraído)</h3>
                    <div class="texto-bruto">{amostra_texto(texto2_bruto)}</div>
                </div>
                
                <div class="texto-container">
                    <h3>🧹 Texto 2 - LIMPO (Após processamento)</h3>
                    <div class="texto-bruto">{amostra_texto(texto2_limpo)}</div>
                </div>
            </div>
        </div>
        
        <div class="warning">
            <p><strong>🔍 O que verificar:</strong></p>
            <ul>
                <li>Se o texto bruto tem pontuações (pontos, vírgulas, etc.)</li>
                <li>Se as quebras de linha estão sendo preservadas</li>
                <li>Se há caracteres especiais que podem estar causando problemas</li>
                <li>Comparar com o resultado dos trechos similares para identificar discrepâncias</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return debug_info

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
