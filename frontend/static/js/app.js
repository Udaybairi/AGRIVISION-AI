/**
 * AGRIVISION AI - Frontend Application Logic & REST Integration
 * "Intelligent Farming. Evidence-Based Decisions."
 */

document.addEventListener('DOMContentLoaded', () => {
  initNavScrollSpy();
  initCropModule();
  initFertilizerModule();
  initDiseaseModule();
  initPestModule();
  initChatModule();
  initObservabilityVisualizer();
});

// Navbar Scroll Spy (Zero-reflow IntersectionObserver implementation)
function initNavScrollSpy() {
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('header[id], section[id], div[id^="module-"]');
  if (!sections.length || !navLinks.length) return;

  const setActiveLink = (id) => {
    navLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (href === `#${id}`) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        setActiveLink(entry.target.id);
      }
    });
  }, {
    rootMargin: '-20% 0px -65% 0px',
    threshold: 0
  });

  sections.forEach(section => observer.observe(section));
}


// ============================================================================
// 1. CROP RECOMMENDATION MODULE
// ============================================================================
function initCropModule() {
  const form = document.getElementById('crop-form');
  const resultBox = document.getElementById('crop-result-box');
  if (!form || !resultBox) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = form.querySelector('button[type="submit"]');
    const origText = btn.innerHTML;
    btn.innerHTML = 'Analyzing Agronomic Profile...';
    btn.disabled = true;

    const payload = {
      nitrogen: parseFloat(document.getElementById('crop-n').value) || 0,
      phosphorus: parseFloat(document.getElementById('crop-p').value) || 0,
      potassium: parseFloat(document.getElementById('crop-k').value) || 0,
      temperature: parseFloat(document.getElementById('crop-temp').value) || 25,
      humidity: parseFloat(document.getElementById('crop-humidity').value) || 70,
      ph: parseFloat(document.getElementById('crop-ph').value) || 6.5,
      rainfall: parseFloat(document.getElementById('crop-rain').value) || 100,
      city: document.getElementById('crop-city').value.trim()
    };

    try {
      const resp = await fetch('/api/crop-recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();

      if (resp.ok) {
        const ml = data.ml_prediction;
        const rag = data.agronomic_evidence;

        document.getElementById('crop-primary-name').innerText = ml.primary_crop;
        document.getElementById('crop-confidence-val').innerText = `${ml.confidence_pct}% Confidence`;
        document.getElementById('crop-profile-desc').innerText = ml.profile.description || 'Optimal agronomic match.';
        document.getElementById('crop-growing-period').innerText = ml.profile.growing_period || 'Standard Cycle';
        document.getElementById('crop-opt-temp').innerText = ml.profile.optimal_temp || `${payload.temperature} °C`;
        document.getElementById('crop-opt-rain').innerText = ml.profile.optimal_rainfall || `${payload.rainfall} mm`;

        // Render alternative crop options
        const altContainer = document.getElementById('crop-alternatives-list');
        altContainer.innerHTML = '';
        if (ml.recommendations && ml.recommendations.length > 1) {
          ml.recommendations.slice(1).forEach(alt => {
            const badge = document.createElement('span');
            badge.className = 'result-badge success';
            badge.innerText = `${alt.crop} (${alt.confidence_pct}%)`;
            altContainer.appendChild(badge);
          });
        }

        // Citations
        renderCitations('crop-citations-box', rag.citations);

        resultBox.style.display = 'block';
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        alert(data.error || 'Failed to predict crop recommendation.');
      }
    } catch (err) {
      console.error(err);
      alert('Network error connecting to crop recommendation service.');
    } finally {
      btn.innerHTML = origText;
      btn.disabled = false;
    }
  });
}


// ============================================================================
// 2. FERTILIZER ADVISORY MODULE
// ============================================================================
function initFertilizerModule() {
  const form = document.getElementById('fert-form');
  const resultBox = document.getElementById('fert-result-box');
  if (!form || !resultBox) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = form.querySelector('button[type="submit"]');
    const origText = btn.innerHTML;
    btn.innerHTML = 'Computing Nutrient Discrepancy...';
    btn.disabled = true;

    const payload = {
      crop: document.getElementById('fert-crop').value,
      nitrogen: parseFloat(document.getElementById('fert-n').value) || 0,
      phosphorus: parseFloat(document.getElementById('fert-p').value) || 0,
      potassium: parseFloat(document.getElementById('fert-k').value) || 0,
      soil_type: document.getElementById('fert-soil').value,
      ph: parseFloat(document.getElementById('fert-ph').value) || null
    };

    try {
      const resp = await fetch('/api/fertilizer-recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();

      if (resp.ok) {
        const adv = data.advisory;
        const rag = data.agronomic_evidence;

        document.getElementById('fert-status-title').innerText = adv.status;
        document.getElementById('fert-symptoms-text').innerText = adv.symptoms;
        document.getElementById('fert-action-text').innerText = adv.recommended_action;
        document.getElementById('fert-organic-text').innerText = adv.organic_alternative;

        // Render nutrient stats
        const nComp = adv.nutrient_comparison;
        document.getElementById('fert-n-stat').innerText = `N: ${nComp.nitrogen.current} (Req: ${nComp.nitrogen.required})`;
        document.getElementById('fert-p-stat').innerText = `P: ${nComp.phosphorus.current} (Req: ${nComp.phosphorus.required})`;
        document.getElementById('fert-k-stat').innerText = `K: ${nComp.potassium.current} (Req: ${nComp.potassium.required})`;

        renderCitations('fert-citations-box', rag.citations);

        resultBox.style.display = 'block';
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        alert(data.error || 'Failed to compute fertilizer advisory.');
      }
    } catch (err) {
      console.error(err);
      alert('Network error connecting to fertilizer advisor.');
    } finally {
      btn.innerHTML = origText;
      btn.disabled = false;
    }
  });
}


// ============================================================================
// 3. PLANT DOCTOR (LEAF DISEASE AI)
// ============================================================================
function initDiseaseModule() {
  const dropzone = document.getElementById('disease-dropzone');
  const fileInput = document.getElementById('disease-file-input');
  const previewBox = document.getElementById('disease-preview-box');
  const previewImg = document.getElementById('disease-preview-img');
  const resultBox = document.getElementById('disease-result-box');
  const analyzeBtn = document.getElementById('disease-analyze-btn');

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewBox.style.display = 'block';
      analyzeBtn.style.display = 'inline-flex';
    };
    reader.readAsDataURL(file);
  }

  analyzeBtn.addEventListener('click', async () => {
    if (!fileInput.files.length) return;

    const origText = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = 'Running Neural Vision Model...';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
      const resp = await fetch('/api/disease-predict', {
        method: 'POST',
        body: formData
      });
      const data = await resp.json();

      if (resp.ok) {
        const cv = data.cv_diagnosis;
        const rag = data.rag_evidence;

        document.getElementById('disease-name-title').innerText = cv.is_healthy ? `Healthy Plant (${cv.crop})` : `${cv.crop} — ${cv.disease}`;
        document.getElementById('disease-confidence-badge').innerText = `${cv.confidence_pct}% AI Vision Confidence`;
        document.getElementById('disease-symptoms-text').innerText = cv.diagnostics.symptoms || 'Visual symptoms matched in knowledge base.';
        document.getElementById('disease-cause-text').innerText = cv.diagnostics.cause || 'Pathogen / physiological factor.';
        document.getElementById('disease-mgmt-text').innerText = cv.diagnostics.management || 'Follow standard agronomic sanitation.';
        document.getElementById('disease-prev-text').innerText = cv.diagnostics.prevention || 'Maintain crop hygiene and certified seeds.';

        // Top Candidates list
        const topList = document.getElementById('disease-candidates-list');
        topList.innerHTML = '';
        if (cv.top_candidates) {
          cv.top_candidates.forEach(c => {
            const span = document.createElement('span');
            span.className = 'result-badge ' + (c.is_healthy ? 'success' : 'warning');
            span.innerText = `${c.disease} (${c.confidence_pct}%)`;
            topList.appendChild(span);
          });
        }

        renderCitations('disease-citations-box', rag.citations);

        resultBox.style.display = 'block';
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        alert(data.error || 'Failed to analyze leaf image.');
      }
    } catch (err) {
      console.error(err);
      alert('Network error connecting to computer vision service.');
    } finally {
      analyzeBtn.innerHTML = origText;
      analyzeBtn.disabled = false;
    }
  });
}


// ============================================================================
// 4. PEST AI MODULE
// ============================================================================
function initPestModule() {
  const form = document.getElementById('pest-form');
  const resultBox = document.getElementById('pest-result-box');
  if (!form || !resultBox) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('pest-query-input').value.trim();
    if (!query) return;

    const btn = form.querySelector('button[type="submit"]');
    const origText = btn.innerHTML;
    btn.innerHTML = 'Scanning Pest Knowledge Base...';
    btn.disabled = true;

    try {
      const resp = await fetch('/api/pest-detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: query })
      });
      const data = await resp.json();

      if (resp.ok) {
        const pest = data.pest_diagnosis;
        const rag = data.rag_evidence;

        document.getElementById('pest-name-title').innerText = pest.identified_pest;
        document.getElementById('pest-scientific-name').innerText = `Scientific Name: ${pest.scientific_name}`;
        document.getElementById('pest-crops-affected').innerText = `Target Crops: ${pest.crops_affected.join(', ')}`;
        document.getElementById('pest-symptoms-text').innerText = pest.symptoms;
        document.getElementById('pest-cultural-text').innerText = pest.ipm_management.cultural_control;
        document.getElementById('pest-bio-text').innerText = pest.ipm_management.biological_control;
        document.getElementById('pest-chem-text').innerText = pest.ipm_management.chemical_safeguards;

        renderCitations('pest-citations-box', rag.citations);

        resultBox.style.display = 'block';
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        alert(data.error || 'Failed to identify pest.');
      }
    } catch (err) {
      console.error(err);
      alert('Network error connecting to pest intelligence service.');
    } finally {
      btn.innerHTML = origText;
      btn.disabled = false;
    }
  });
}


// ============================================================================
// 5. ADVANCED RAG CONVERSATIONAL ASSISTANT
// ============================================================================
function initChatModule() {
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const messagesContainer = document.getElementById('chat-messages');
  if (!form || !input || !messagesContainer) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    // Append User Message
    appendChatMessage('user', query);
    input.value = '';

    // Create Assistant Loading Placeholder
    const loadingMessageId = 'loading-' + Date.now();
    appendLoadingMessage(loadingMessageId);

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
      });
      const data = await resp.json();

      removeLoadingMessage(loadingMessageId);

      if (resp.ok) {
        appendAssistantRAGResponse(data);
      } else {
        appendChatMessage('assistant', `⚠️ ${data.error || 'Error processing inquiry.'}`);
      }
    } catch (err) {
      console.error(err);
      removeLoadingMessage(loadingMessageId);
      appendChatMessage('assistant', '⚠️ Network communication error. Please retry.');
    }
  });

  function appendChatMessage(sender, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${sender}`;
    msgDiv.innerHTML = `
      <div class="message-avatar">${sender === 'user' ? '👨‍🌾' : '🌿'}</div>
      <div class="message-bubble">${escapeHtml(text)}</div>
    `;
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function appendLoadingMessage(id) {
    const msgDiv = document.createElement('div');
    msgDiv.id = id;
    msgDiv.className = 'chat-message assistant';
    msgDiv.innerHTML = `
      <div class="message-avatar">🌿</div>
      <div class="message-bubble" style="display: flex; gap: 8px; align-items: center;">
        <span class="pulse-dot"></span>
        <span style="font-size: 0.88rem; color: var(--text-secondary);">Query Rewriter → Hybrid Search → Reranking Knowledge Base...</span>
      </div>
    `;
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function removeLoadingMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function appendAssistantRAGResponse(data) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-message assistant';

    // 1. Farmer-Friendly Query Understanding Header
    let understandingHTML = '';
    if (data.query_transformation) {
      const qMeta = data.query_transformation.metadata || {};
      const cropVal = qMeta.crop || data.crop?.name || 'General Crop';
      const categoryVal = data.category || data.intent || 'Agronomic Consultation';
      understandingHTML = `
        <div class="query-understanding-badge" style="margin-bottom: 12px;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
            <span>🌱 Target: <strong>${escapeHtml(cropVal)}</strong></span>
            <span class="structured-card-badge" style="background: rgba(16, 185, 129, 0.15); color: var(--emerald-light);">
              ${escapeHtml(categoryVal.replace(/_/g, ' '))}
            </span>
          </div>
        </div>
      `;
    }

    // 2. Parse response (JSON or Sectioned Markdown) into Structured Cards
    let cardsHTML = '';
    let parsedJSON = null;

    // Check if data.answer is already a JSON string or object
    if (typeof data.answer === 'object' && data.answer !== null && data.diagnosis) {
      parsedJSON = data.answer;
    } else if (typeof data.answer === 'string') {
      const trimmed = data.answer.trim();
      if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
        try {
          parsedJSON = JSON.parse(trimmed);
        } catch (e) {
          parsedJSON = null;
        }
      }
    }

    if (parsedJSON) {
      cardsHTML = renderJSONStructuredCards(parsedJSON, data);
    } else {
      cardsHTML = renderMarkdownStructuredCards(data.answer || '', data);
    }

    // 3. Visual Gallery Evidence
    let visualHTML = '';
    const visualList = (data.visual_evidence && data.visual_evidence.length > 0) 
      ? data.visual_evidence 
      : (parsedJSON && parsedJSON.images ? parsedJSON.images.map(img => ({
          image_url: img.url,
          label: img.label || 'Reference Image',
          category: img.type || 'Visual Reference'
        })) : []);

    if (visualList.length > 0) {
      const imgCards = visualList.map(img => `
        <div class="visual-card" onclick="openImageModal('${img.image_url}', '${escapeHtml(img.label)}')">
          <div class="visual-img-wrap">
            <img src="${img.image_url}" alt="${escapeHtml(img.label)}" loading="lazy" />
          </div>
          <div class="visual-card-info">
            <span class="visual-tag">${escapeHtml(img.category || 'Reference')}</span>
            <span class="visual-label">${escapeHtml(img.label || 'Visual Reference')}</span>
          </div>
        </div>
      `).join('');

      visualHTML = `
        <div class="visual-gallery-container">
          <div class="visual-gallery-title">
            <span>🖼️ Visual Reference Library (${visualList.length})</span>
          </div>
          <div class="visual-gallery-grid">
            ${imgCards}
          </div>
        </div>
      `;
    }

    // 4. Compact Collapsible Sources / Citations
    let citationsHTML = '';
    const citationsList = (data.evidence_items && data.evidence_items.length > 0)
      ? data.evidence_items
      : (parsedJSON && parsedJSON.citations ? parsedJSON.citations.map(c => ({
          citation_tag: `[${c.id || 'Source'}]`,
          source: c.title || 'Agronomy Knowledge Base',
          snippet: c.page ? `Reference: ${c.page}` : 'Verified Agriculture Dataset',
          relevance_pct: c.relevance ? Math.round(c.relevance * 100) : 90
        })) : []);

    if (citationsList.length > 0) {
      const citItems = citationsList.map(c => `
        <div class="source-item">
          <div class="source-header">
            <span class="source-tag">${escapeHtml(c.citation_tag)} ${escapeHtml(c.source)}</span>
            ${c.relevance_pct ? `<span class="source-relevance">${c.relevance_pct}% Match</span>` : ''}
          </div>
          <div style="color: var(--text-muted); font-size: 0.78rem; margin-top: 4px;">${escapeHtml(c.snippet || '')}</div>
        </div>
      `).join('');

      citationsHTML = `
        <div class="sources-panel">
          <button class="sources-toggle-btn" onclick="toggleSources(this)">
            <span>📚 Verified Sources (${citationsList.length})</span>
            <span style="font-size: 0.7rem;">▼</span>
          </button>
          <div class="sources-list" style="display: none;">
            ${citItems}
          </div>
        </div>
      `;
    }

    msgDiv.innerHTML = `
      <div class="message-avatar">🌿</div>
      <div class="message-bubble 3d-glass" style="width: 100%;">
        ${understandingHTML}
        <div class="structured-response-wrapper">
          ${cardsHTML}
        </div>
        ${visualHTML}
        ${citationsHTML}
      </div>
    `;

    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Renderer for Structured JSON format
  function renderJSONStructuredCards(json, rawData) {
    let out = '';

    // 1. Short Summary
    const summaryText = json.problem?.summary || json.answer?.short_answer || (typeof json.answer === 'string' ? json.answer : '');
    if (summaryText) {
      out += `
        <div class="structured-summary-box">
          ${escapeHtml(summaryText)}
        </div>
      `;
    }

    // 2. Diagnosis / Problem Card
    const prob = json.problem || json.diagnosis?.primary;
    if (prob) {
      const probName = prob.name || 'Diagnosed Condition';
      const sciName = prob.scientific_name || '';
      const confRaw = String(prob.confidence || 'Medium');
      const confLevel = confRaw.toLowerCase().includes('high') ? 'high' : (confRaw.toLowerCase().includes('low') ? 'low' : 'medium');
      const confBadgeText = /^\d+(\.\d+)?$/.test(confRaw) ? `${Math.round(parseFloat(confRaw) * 100)}% Match` : `${confRaw} Confidence`;

      const symptomsList = json.symptoms || json.diagnosis?.symptoms || [];
      let symptomsHTML = '';
      if (symptomsList.length > 0) {
        symptomsHTML = `
          <div style="margin-top: 10px;">
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 6px; font-weight: 600;">Key Observed Symptoms:</div>
            <ul class="structured-list">
              ${symptomsList.map(s => `
                <li class="structured-list-item">
                  <span class="item-icon">🔍</span>
                  <span>${escapeHtml(s)}</span>
                </li>
              `).join('')}
            </ul>
          </div>
        `;
      }

      out += `
        <div class="structured-card diagnosis">
          <div class="structured-card-header">
            <div class="structured-card-title">
              <span>🌱 Diagnosis:</span>
              <span style="color: var(--emerald-light);">${escapeHtml(probName)}</span>
              ${sciName ? `<em style="font-size: 0.8rem; color: var(--text-muted); font-weight: 400;">(${escapeHtml(sciName)})</em>` : ''}
            </div>
            <span class="structured-card-badge ${confLevel}">${escapeHtml(confBadgeText)}</span>
          </div>
          ${symptomsHTML}
        </div>
      `;
    }

    // 3. Immediate Action Checklist
    const actionList = json.what_to_do_now || json.answer?.what_to_do_now || [];
    if (actionList.length > 0) {
      out += `
        <div class="structured-card action">
          <div class="structured-card-header">
            <div class="structured-card-title">
              <span>🩺 What to Do Now (Immediate Checklist)</span>
            </div>
          </div>
          <ul class="structured-list">
            ${actionList.map(item => `
              <li class="structured-list-item">
                <span class="item-icon" style="color: var(--cyan-accent);">✓</span>
                <span>${escapeHtml(item)}</span>
              </li>
            `).join('')}
          </ul>
        </div>
      `;
    }

    // 4. Treatment & Medicine Card
    let treatmentList = [];
    if (Array.isArray(json.treatment)) {
      treatmentList = json.treatment;
    } else if (json.treatment && typeof json.treatment === 'object') {
      treatmentList = json.treatment.recommended || [];
    }

    const safetyList = json.safety || json.treatment?.safety || [
      "Wear appropriate protective equipment when spraying.",
      "Do not exceed the label-recommended dose.",
      "Follow the required pre-harvest interval."
    ];

    if (treatmentList.length > 0) {
      const renderMeds = (list) => list.map(m => `
        <div class="structured-medicine-box">
          <div class="medicine-name">
            <span>💊 ${escapeHtml(m.name || 'Recommended Treatment')}</span>
            ${m.dosage ? `<span class="medicine-tag dosage">Dosage: ${escapeHtml(m.dosage)}</span>` : ''}
          </div>
          <div class="medicine-meta">
            ${m.type ? `<span class="medicine-tag">${escapeHtml(m.type)}</span>` : ''}
            ${m.instruction ? `<span style="font-size: 0.78rem; color: var(--text-secondary);">${escapeHtml(m.instruction)}</span>` : ''}
            ${m.purpose ? `<span style="font-size: 0.78rem; color: var(--text-secondary);">${escapeHtml(m.purpose)}</span>` : ''}
          </div>
        </div>
      `).join('');

      out += `
        <div class="structured-card treatment">
          <div class="structured-card-header">
            <div class="structured-card-title">
              <span>💊 Recommended Treatment & Medicines</span>
            </div>
          </div>
          ${renderMeds(treatmentList)}
          <div class="safety-alert-box">
            <span>⚠️</span>
            <div>
              <strong>Safety Precautions:</strong> ${escapeHtml(safetyList.join(' '))}
            </div>
          </div>
        </div>
      `;
    }

    // 5. Prevention Protocol
    const prevList = json.prevention || [];
    if (prevList.length > 0) {
      out += `
        <div class="structured-card prevention">
          <div class="structured-card-header">
            <div class="structured-card-title">
              <span>🛡️ Long-term Prevention Protocol</span>
            </div>
          </div>
          <ul class="structured-list">
            ${prevList.map(p => `
              <li class="structured-list-item">
                <span class="item-icon" style="color: var(--emerald-light);">🛡️</span>
                <span>${escapeHtml(p)}</span>
              </li>
            `).join('')}
          </ul>
        </div>
      `;
    }

    // 6. Follow-Up Question
    const followUpText = (typeof json.follow_up === 'string') ? json.follow_up : json.follow_up?.question;
    if (followUpText) {
      out += `
        <div class="structured-card followup">
          <div class="structured-card-header">
            <div class="structured-card-title">
              <span>❓ Follow-Up Recommendation</span>
            </div>
          </div>
          <div class="followup-prompt">${escapeHtml(followUpText)}</div>
          <button class="followup-action-btn" onclick="document.getElementById('module-disease').scrollIntoView({behavior: 'smooth'})">
            <span>📷 Open Plant Doctor Module</span>
            <span>→</span>
          </button>
        </div>
      `;
    }

    return out;
  }

  // Renderer for Markdown Sectioned text (extracts sections into cards)
  function renderMarkdownStructuredCards(rawText, rawData) {
    if (!rawText) return '<div class="structured-summary-box">No response content received.</div>';

    // Normalize headers (handles `## Header`, `**## Header**`, `### Header`)
    const lines = rawText.split('\n');
    const sections = [];
    let currentSection = { title: 'Summary', icon: '📝', type: 'summary', content: [] };

    for (let line of lines) {
      const cleanLine = line.trim().replace(/^\*\*|\*\*$/g, '').trim();
      const headerMatch = cleanLine.match(/^#{2,4}\s*(.*)$/);
      if (headerMatch) {
        if (currentSection.content.length > 0) {
          sections.push(currentSection);
        }
        const hText = headerMatch[1].replace(/\*\*/g, '').trim();
        let sType = 'default';
        let sIcon = '📌';

        if (/diagnosis|issue|likely/i.test(hText)) {
          sType = 'diagnosis'; sIcon = '🌱';
        } else if (/what to do|action|step|remedy/i.test(hText)) {
          sType = 'action'; sIcon = '🩺';
        } else if (/treatment|medicine|chemical|fungicide|pesticide/i.test(hText)) {
          sType = 'treatment'; sIcon = '💊';
        } else if (/prevention|prevent|long-term/i.test(hText)) {
          sType = 'prevention'; sIcon = '🛡️';
        } else if (/why|cause|scientific/i.test(hText)) {
          sType = 'diagnosis'; sIcon = '🔍';
        } else if (/important|precaution|safety|warning/i.test(hText)) {
          sType = 'safety'; sIcon = '⚠️';
        }

        currentSection = { title: hText.replace(/^[🌱🔍🩺🛡️⚠️💊📚📝📌\s]+/, ''), icon: sIcon, type: sType, content: [] };
      } else {
        currentSection.content.push(line);
      }
    }
    if (currentSection.content.length > 0) {
      sections.push(currentSection);
    }

    let out = '';
    sections.forEach((sec, idx) => {
      let textBlock = sec.content.join('\n').trim();
      if (!textBlock) return;

      // Split text on inline numbered steps if present e.g. "1. Scout... 2. Cultural..."
      textBlock = textBlock.replace(/\s+(\d+\.\s+)/g, '\n$1');

      // Filter out markdown table separator lines |---|---|
      textBlock = textBlock.replace(/\|[-|\s]+\|/g, '');

      // Extract bullet lines or split numbered / dash lines
      let rawBulletLines = [];
      const textLines = textBlock.split('\n');
      textLines.forEach(l => {
        const tr = l.trim();
        if (!tr) return;
        if (tr.startsWith('-') || tr.startsWith('*') || /^\d+\./.test(tr) || tr.startsWith('•')) {
          rawBulletLines.push(tr);
        } else if (tr.startsWith('|') && tr.endsWith('|') && !tr.includes('---')) {
          const cells = tr.split('|').map(c => c.trim()).filter(Boolean);
          if (cells.length > 0) {
            rawBulletLines.push(cells.join(' — '));
          }
        } else if (textLines.length > 1 && tr.length < 250) {
          rawBulletLines.push(tr);
        }
      });

      const isList = rawBulletLines.length > 0;

      let formattedBody = '';
      if (isList) {
        formattedBody = `
          <ul class="structured-list">
            ${rawBulletLines.map(b => {
              let cleaned = b.replace(/^[-*•\d.]+\s*/, '').trim();
              
              // If line has "Title — Details", format title bold
              if (cleaned.includes(' — ')) {
                const parts = cleaned.split(' — ');
                cleaned = `<strong>${parts[0].trim()}</strong>: ${parts.slice(1).join(' — ').trim()}`;
              }

              const highlighted = cleaned.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
              let bulletIcon = '•';
              if (sec.type === 'action') bulletIcon = '✓';
              else if (sec.type === 'prevention') bulletIcon = '🛡️';
              else if (sec.type === 'treatment') bulletIcon = '💊';
              else if (sec.type === 'diagnosis') bulletIcon = '🔍';

              return `
                <li class="structured-list-item">
                  <span class="item-icon" style="color: var(--emerald-light);">${bulletIcon}</span>
                  <span>${highlighted}</span>
                </li>
              `;
            }).join('')}
          </ul>
        `;
      } else {
        formattedBody = `<div>${textBlock.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n\n/g, '<p style="margin: 6px 0;"></p>')}</div>`;
      }

      if (idx === 0 && (sec.type === 'summary' || /understanding|summary/i.test(sec.title))) {
        out += `<div class="structured-summary-box">${formattedBody}</div>`;
      } else if (sec.type === 'safety') {
        out += `
          <div class="safety-alert-box" style="margin-top: 10px;">
            <span>⚠️</span>
            <div><strong>${escapeHtml(sec.title)}:</strong> ${formattedBody}</div>
          </div>
        `;
      } else {
        out += `
          <div class="structured-card ${sec.type}">
            <div class="structured-card-header">
              <div class="structured-card-title">
                <span>${sec.icon} ${escapeHtml(sec.title)}</span>
              </div>
            </div>
            ${formattedBody}
          </div>
        `;
      }
    });

    return out;
  }
}

window.toggleSources = function(btn) {
  const list = btn.nextElementSibling;
  if (list.style.display === 'none') {
    list.style.display = 'flex';
    btn.querySelector('span:last-child').innerText = '▲';
  } else {
    list.style.display = 'none';
    btn.querySelector('span:last-child').innerText = '▼';
  }
};

window.openImageModal = function(src, label) {
  const existing = document.getElementById('agri-image-modal');
  if (existing) existing.remove();

  const modal = document.createElement('div');
  modal.id = 'agri-image-modal';
  modal.className = 'image-modal-backdrop';
  modal.onclick = (e) => {
    if (e.target === modal || e.target.classList.contains('image-modal-close')) {
      modal.remove();
    }
  };

  modal.innerHTML = `
    <div class="image-modal-content" onclick="event.stopPropagation()">
      <div class="image-modal-header">
        <div class="image-modal-title">🔍 Visual Reference: ${escapeHtml(label)}</div>
        <button class="image-modal-close" onclick="document.getElementById('agri-image-modal').remove()">✕</button>
      </div>
      <div class="image-modal-body">
        <img src="${src}" alt="${escapeHtml(label)}" />
      </div>
    </div>
  `;
  document.body.appendChild(modal);
};


// ============================================================================
// 6. RAG OBSERVABILITY & TRACE VISUALIZER
// ============================================================================
function initObservabilityVisualizer() {
  const triggerBtn = document.getElementById('obs-run-btn');
  if (!triggerBtn) return;

  triggerBtn.addEventListener('click', async () => {
    const inputQ = document.getElementById('obs-query-input').value.trim() || 'tomato leaf black what medicine';
    triggerBtn.disabled = true;
    triggerBtn.innerText = 'Tracing Pipeline...';

    try {
      const resp = await fetch(`/api/observability?query=${encodeURIComponent(inputQ)}`);
      const data = await resp.json();

      if (resp.ok) {
        document.getElementById('obs-raw-q').innerText = data.sample_query;
        document.getElementById('obs-rewrite-q').innerText = data.query_transformation.rewritten_query;
        document.getElementById('obs-category').innerText = data.category;
        document.getElementById('obs-latency').innerText = `${data.latency_ms} ms`;
        document.getElementById('obs-grounding').innerText = `${data.grounding_level} (${Math.round(data.grounding_score * 100)}%)`;

        // Highlight active flow nodes
        document.querySelectorAll('.rag-node').forEach(node => {
          node.classList.add('active');
        });
      }
    } catch (err) {
      console.error(err);
    } finally {
      triggerBtn.disabled = false;
      triggerBtn.innerText = 'Run Pipeline Trace';
    }
  });
}


window.quickFillChat = function(query) {
  const chatInput = document.getElementById('chat-input');
  const chatSection = document.getElementById('module-chat');
  if (chatInput) {
    chatInput.value = query;
    if (chatSection) {
      chatSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    setTimeout(() => {
      chatInput.focus();
      chatInput.style.borderColor = 'var(--emerald-primary)';
      chatInput.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.4)';
      setTimeout(() => {
        chatInput.style.borderColor = '';
        chatInput.style.boxShadow = '';
      }, 1800);
    }, 400);
  }
};

// Helper: Citations box renderer
function renderCitations(containerId, citations) {
  const container = document.getElementById(containerId);
  if (!container || !citations || !citations.length) return;

  container.innerHTML = `
    <h5 style="color: var(--text-secondary); margin-bottom: 8px; font-size: 0.85rem;">Evidence Citations:</h5>
    <div style="display: flex; flex-direction: column; gap: 6px;">
      ${citations.map(c => `
        <div class="source-item">
          <span class="source-tag">${c.citation_tag} ${escapeHtml(c.source)}</span>
          <span class="source-relevance"> (${c.relevance_pct || 90}% Match)</span>
        </div>
      `).join('')}
    </div>
  `;
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

