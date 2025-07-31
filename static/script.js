function showTab(tabName) {
  // Esconder todas as abas
  document.querySelectorAll('.tab-content').forEach(tab => {
    tab.classList.remove('active');
  });
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.remove('active');
  });
  
  // Mostrar aba selecionada
  document.getElementById(tabName).classList.add('active');
  document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
}

// Funções para drag and drop
function setupDragDrop(dropAreaId, inputId) {
  const dropArea = document.getElementById(dropAreaId);
  const fileInput = document.getElementById(inputId);
  
  if (!dropArea || !fileInput) return;
  
  // Prevenir comportamento padrão
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });
  
  // Destacar área quando arquivo está sobre ela
  ['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
  });
  
  ['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
  });
  
  // Manipular arquivos soltos
  dropArea.addEventListener('drop', handleDrop, false);
  
  // Click para abrir seletor de arquivo
  dropArea.addEventListener('click', () => fileInput.click());
  
  // Atualizar quando arquivo é selecionado via input
  fileInput.addEventListener('change', handleFileSelect);
  
  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }
  
  function highlight(e) {
    dropArea.classList.add('dragover');
  }
  
  function unhighlight(e) {
    dropArea.classList.remove('dragover');
  }
  
  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
      fileInput.files = files;
      updateDropArea(dropArea, files[0]);
    }
  }
  
  function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
      updateDropArea(dropArea, files[0]);
    }
  }
  
  function updateDropArea(area, file) {
    area.classList.add('file-selected');
    area.innerHTML = `
      <div class="drag-drop-icon">📄</div>
      <div class="file-name">${file.name}</div>
      <div class="drag-drop-text">Arquivo selecionado (${(file.size / 1024).toFixed(1)} KB)</div>
      <button type="button" class="remove-file" onclick="clearFile('${area.id}', '${fileInput.id}')">Remover</button>
    `;
  }
}

function clearFile(dropAreaId, inputId) {
  const dropArea = document.getElementById(dropAreaId);
  const fileInput = document.getElementById(inputId);
  
  fileInput.value = '';
  dropArea.classList.remove('file-selected');
  
  // Restaurar conteúdo original baseado no ID
  if (dropAreaId.includes('individual')) {
    dropArea.innerHTML = `
      <div class="drag-drop-icon">📁</div>
      <div class="drag-drop-text">Arraste um arquivo aqui ou clique para selecionar</div>
      <small>Arquivos aceitos: PDF, TXT</small>
    `;
  } else {
    const fileNumber = dropAreaId.includes('1') ? '1' : '2';
    dropArea.innerHTML = `
      <div class="drag-drop-icon">📁</div>
      <div class="drag-drop-text">Arraste o arquivo ${fileNumber} aqui ou clique para selecionar</div>
      <small>Arquivos aceitos: PDF, TXT</small>
    `;
  }
}

// Sistema de Loading com Progresso
const LoadingManager = {
  overlay: null,
  progressBar: null,
  progressText: null,
  progressDetails: null,
  steps: [],
  currentStep: 0,
  
  init() {
    this.overlay = document.getElementById('loadingOverlay');
    this.progressBar = this.overlay.querySelector('.progress-fill');
    this.progressText = this.overlay.querySelector('.progress-text');
    this.progressDetails = this.overlay.querySelector('.progress-details');
    this.steps = this.overlay.querySelectorAll('.step-item');
  },
  
  show() {
    if (this.overlay) {
      this.overlay.classList.add('show');
      this.reset();
    }
  },
  
  hide() {
    if (this.overlay) {
      this.overlay.classList.remove('show');
    }
  },
  
  reset() {
    this.currentStep = 0;
    this.updateProgress(0, 'Iniciando processamento...', 'Preparando análise dos arquivos');
    this.updateSteps();
  },
  
  updateProgress(percentage, text, details) {
    if (this.progressBar) this.progressBar.style.width = percentage + '%';
    if (this.progressText) this.progressText.textContent = percentage + '% Completo';
    if (this.progressDetails) this.progressDetails.textContent = details || text;
  },
  
  updateSteps() {
    this.steps.forEach((step, index) => {
      const icon = step.querySelector('.step-icon');
      
      if (index < this.currentStep) {
        // Etapa concluída
        step.className = 'step-item step-completed';
        icon.innerHTML = '<i class="fas fa-check"></i>';
      } else if (index === this.currentStep) {
        // Etapa atual
        step.className = 'step-item step-current';
        icon.textContent = index + 1;
      } else {
        // Etapa pendente
        step.className = 'step-item step-pending';
        icon.textContent = index + 1;
      }
    });
  },
  
  nextStep(text, details) {
    this.currentStep++;
    this.updateSteps();
    
    const percentage = Math.round((this.currentStep / this.steps.length) * 100);
    this.updateProgress(percentage, text, details);
  },
  
  // Simular progresso automático para demonstração
  simulateProgress(steps) {
    let currentIndex = 0;
    
    const processStep = () => {
      if (currentIndex < steps.length) {
        const step = steps[currentIndex];
        this.updateProgress(step.percentage, step.text, step.details);
        
        if (step.nextStep) {
          this.nextStep(step.text, step.details);
        }
        
        currentIndex++;
        setTimeout(processStep, step.duration || 1000);
      }
    };
    
    processStep();
  }
};

// Funções para controlar o modal de informações sobre métricas
function openMetricsModal() {
  const modal = document.getElementById('metricsModal');
  if (modal) {
    modal.classList.add('show');
    document.body.style.overflow = 'hidden'; // Prevenir scroll da página
  }
}

function closeMetricsModal() {
  const modal = document.getElementById('metricsModal');
  if (modal) {
    modal.classList.remove('show');
    document.body.style.overflow = 'auto'; // Restaurar scroll da página
  }
}

// Fechar modal com tecla ESC
document.addEventListener('keydown', function(event) {
  if (event.key === 'Escape') {
    closeMetricsModal();
  }
});

// Interceptar envio de formulários para mostrar loading
function setupFormSubmission() {
  const forms = document.querySelectorAll('form[method="post"]');
  
  forms.forEach(form => {
    form.addEventListener('submit', function(e) {
      const mode = this.querySelector('input[name="modo"]')?.value || 'individual';
      
      // Verificar se arquivos foram selecionados
      const fileInputs = this.querySelectorAll('input[type="file"]');
      let hasFiles = true;
      
      fileInputs.forEach(input => {
        if (input.required && !input.files.length) {
          hasFiles = false;
        }
      });
      
      if (!hasFiles) {
        return; // Deixar validação padrão do HTML5 funcionar
      }
      
      // Mostrar loading
      LoadingManager.show();
      
      // Definir etapas baseadas no modo
      if (mode === 'comparar') {
        LoadingManager.simulateProgress([
          { percentage: 10, text: 'Carregando arquivos...', details: 'Lendo conteúdo dos PDFs/TXTs', duration: 800 },
          { percentage: 25, text: 'Extraindo texto...', details: 'Processando e limpando o conteúdo', nextStep: true, duration: 1200 },
          { percentage: 45, text: 'Calculando similaridade...', details: 'Aplicando algoritmos TF-IDF e cosine similarity', nextStep: true, duration: 1500 },
          { percentage: 65, text: 'Analisando IA...', details: 'Executando 7 métricas de detecção de IA', nextStep: true, duration: 1800 },
          { percentage: 85, text: 'Encontrando trechos similares...', details: 'Identificando seções com maior correlação', duration: 1000 },
          { percentage: 100, text: 'Finalizando...', details: 'Preparando resultados para exibição', duration: 500 }
        ]);
      } else {
        LoadingManager.simulateProgress([
          { percentage: 15, text: 'Carregando arquivo...', details: 'Lendo conteúdo do documento', duration: 800 },
          { percentage: 35, text: 'Extraindo texto...', details: 'Processando e limpando o conteúdo', nextStep: true, duration: 1000 },
          { percentage: 70, text: 'Analisando IA...', details: 'Executando análise avançada com 7 métricas', nextStep: true, duration: 2000 },
          { percentage: 100, text: 'Finalizando...', details: 'Preparando resultados da análise', duration: 500 }
        ]);
      }
    });
  });
}

// Inicializar tudo quando a página carregar
document.addEventListener('DOMContentLoaded', function() {
  setupDragDrop('dropArea1', 'arquivo1');
  setupDragDrop('dropArea2', 'arquivo2');
  setupDragDrop('dropAreaIndividual', 'arquivo');
  
  LoadingManager.init();
  setupFormSubmission();
});

// Função para expandir/contrair texto dos trechos similares
function toggleText(textNumber, itemIndex) {
  const truncatedElement = document.getElementById(`text${textNumber}-truncated-${itemIndex}`);
  const fullElement = document.getElementById(`text${textNumber}-full-${itemIndex}`);
  const button = event.target.closest('.expand-btn');
  
  if (!truncatedElement || !fullElement || !button) return;
  
  const expandText = button.querySelector('.expand-text');
  const collapseText = button.querySelector('.collapse-text');
  const expandIcon = button.querySelector('.expand-icon');
  const collapseIcon = button.querySelector('.collapse-icon');
  
  // Verificar estado atual
  const isExpanded = fullElement.style.display !== 'none';
  
  if (isExpanded) {
    // Contrair
    truncatedElement.style.display = 'block';
    fullElement.style.display = 'none';
    expandText.style.display = 'inline';
    collapseText.style.display = 'none';
    expandIcon.style.display = 'inline';
    collapseIcon.style.display = 'none';
    button.classList.remove('expanded');
  } else {
    // Expandir
    truncatedElement.style.display = 'none';
    fullElement.style.display = 'block';
    expandText.style.display = 'none';
    collapseText.style.display = 'inline';
    expandIcon.style.display = 'none';
    collapseIcon.style.display = 'inline';
    button.classList.add('expanded');
  }
}