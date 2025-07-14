# 🔍 Verificador de Texto

## 📋 O que faz

Aplicação web que analisa textos para:
- **Detectar se foi gerado por IA** (ChatGPT, etc.)
- **Comparar similaridade entre dois documentos**
- Suporte para arquivos PDF e TXT
- Gera relatórios em PDF

## 🚀 Como executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar NLTK (apenas na primeira vez)
```python
import nltk
nltk.download('stopwords')
```

### 3. Executar
```bash
python app.py
```

### 4. Acessar
Abra o navegador em: `http://localhost:5000`

## 📖 Como usar

**Análise de IA:** Envie um arquivo → Veja a probabilidade de ser gerado por IA

**Comparação:** Envie dois arquivos → Veja o índice de similaridade entre eles

---
*Desenvolvido com Python + Flask*