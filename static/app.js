/**
 * RMS TITANIC — ML PREDICTOR & DEATH COUNT SIMULATOR
 * Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    activeTab: 'tab-predictor',
    currentModel: 'svc',
    models: {},
    manifestPage: 0,
    manifestLimit: 50,
    manifestSearch: '',
    customCohorts: [
      { id: 1, label: '3rd Class Men', pclass: 3, sex: 'male', age: 26, count: 120, fare: 8, sibsp: 0, parch: 0, embarked: 'S' },
      { id: 2, label: '1st Class Women', pclass: 1, sex: 'female', age: 32, count: 60, fare: 90, sibsp: 1, parch: 0, embarked: 'C' },
      { id: 3, label: 'Steerage Children', pclass: 3, sex: 'male', age: 7, count: 35, fare: 15, sibsp: 2, parch: 1, embarked: 'S' }
    ]
  };

  // DOM Elements
  const elements = {
    // Navigation
    navTabs: document.querySelectorAll('.nav-tab-btn'),
    tabPanes: document.querySelectorAll('.tab-pane'),
    activeModelHeader: document.getElementById('activeModelNameHeader'),
    
    // Passenger Predictor Controls
    archetypeSelect: document.getElementById('archetypeSelect'),
    modelSelect: document.getElementById('modelSelect'),
    pclassSelector: document.getElementById('pclassSelector'),
    pclassInput: document.getElementById('pclassInput'),
    genderSelector: document.getElementById('genderSelector'),
    sexInput: document.getElementById('sexInput'),
    ageSlider: document.getElementById('ageSlider'),
    ageNumber: document.getElementById('ageNumber'),
    ageDisplay: document.getElementById('ageDisplay'),
    sibspInput: document.getElementById('sibspInput'),
    parchInput: document.getElementById('parchInput'),
    fareSlider: document.getElementById('fareSlider'),
    fareInput: document.getElementById('fareInput'),
    fareDisplay: document.getElementById('fareDisplay'),
    embarkedSelect: document.getElementById('embarkedSelect'),
    predictBtn: document.getElementById('predictBtn'),
    
    // Prediction Display
    statusPulse: document.getElementById('statusPulse'),
    statusText: document.getElementById('statusText'),
    gaugeProgress: document.getElementById('gaugeProgress'),
    survivalPctDisplay: document.getElementById('survivalPctDisplay'),
    mainOutcomeBadge: document.getElementById('mainOutcomeBadge'),
    outcomeIcon: document.getElementById('outcomeIcon'),
    outcomeLabel: document.getElementById('outcomeLabel'),
    deathPctDisplay: document.getElementById('deathPctDisplay'),
    confidenceDisplay: document.getElementById('confidenceDisplay'),
    
    // Realistic Boarding Ticket Elements
    ticketPassengerName: document.getElementById('ticketPassengerName'),
    ticketNumberDisplay: document.getElementById('ticketNumberDisplay'),
    ticketStubSerial: document.getElementById('ticketStubSerial'),
    tClassDisplay: document.getElementById('tClassDisplay'),
    ticketBerthDisplay: document.getElementById('ticketBerthDisplay'),
    tPortDisplay: document.getElementById('tPortDisplay'),
    tDemographicsDisplay: document.getElementById('tDemographicsDisplay'),
    tFareDisplay: document.getElementById('tFareDisplay'),
    ticketStamp: document.getElementById('ticketStamp'),
    ticketStampText: document.getElementById('ticketStampText'),
    ticketStampSub: document.getElementById('ticketStampSub'),
    ticketStampDate: document.getElementById('ticketStampDate'),
    factorsList: document.getElementById('factorsList'),
    
    // Death Count Manifest Banner
    manifestClassFilter: document.getElementById('manifestClassFilter'),
    manifestSexFilter: document.getElementById('manifestSexFilter'),
    refreshManifestStatsBtn: document.getElementById('refreshManifestStatsBtn'),
    cntTotalPassengers: document.getElementById('cntTotalPassengers'),
    cntPredictedDeaths: document.getElementById('cntPredictedDeaths'),
    cntActualDeathsSub: document.getElementById('cntActualDeathsSub'),
    cntPredictedSurvivors: document.getElementById('cntPredictedSurvivors'),
    cntActualSurvivorsSub: document.getElementById('cntActualSurvivorsSub'),
    cntModelAccuracy: document.getElementById('cntModelAccuracy'),
    cntDeathErrorSub: document.getElementById('cntDeathErrorSub'),
    classBreakdownList: document.getElementById('classBreakdownList'),
    genderBreakdownList: document.getElementById('genderBreakdownList'),
    ageBreakdownList: document.getElementById('ageBreakdownList'),

    // Custom Simulator
    cohortsContainer: document.getElementById('cohortsContainer'),
    addCohortBtn: document.getElementById('addCohortBtn'),
    runCustomSimBtn: document.getElementById('runCustomSimBtn'),
    simResultSummary: document.getElementById('simResultSummary'),
    simTotalPeople: document.getElementById('simTotalPeople'),
    simTotalDeaths: document.getElementById('simTotalDeaths'),
    simTotalSurvivors: document.getElementById('simTotalSurvivors'),
    simMortalityRate: document.getElementById('simMortalityRate'),

    // CSV Upload
    csvDropzone: document.getElementById('csvDropzone'),
    csvFileInput: document.getElementById('csvFileInput'),
    csvResultBox: document.getElementById('csvResultBox'),
    csvFileName: document.getElementById('csvFileName'),
    csvDeaths: document.getElementById('csvDeaths'),
    csvSurvivors: document.getElementById('csvSurvivors'),
    csvTableBody: document.getElementById('csvTableBody'),

    // Models Benchmark
    modelsBenchmarkGrid: document.getElementById('modelsBenchmarkGrid'),
    cmDisplayGrid: document.getElementById('cmDisplayGrid'),

    // Historical Manifest
    manifestSearchInput: document.getElementById('manifestSearchInput'),
    historicalTableBody: document.getElementById('historicalTableBody'),
    prevManifestPage: document.getElementById('prevManifestPage'),
    nextManifestPage: document.getElementById('nextManifestPage'),
    manifestPageIndicator: document.getElementById('manifestPageIndicator')
  };

  // Initialize App
  init();

  async function init() {
    setupTabNavigation();
    setupPredictorControls();
    setupCustomSimUI();
    setupCSVUpload();
    setupManifestSearch();
    
    // Load initial data from APIs
    await loadModelsData();
    await triggerSinglePrediction();
    await loadManifestDeathCountStats();
    await loadHistoricalManifest();
  }

  /* -------------------------------------------------------------
     1. TAB NAVIGATION
     ------------------------------------------------------------- */
  function setupTabNavigation() {
    elements.navTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const targetId = tab.dataset.target;
        switchTab(targetId);
        window.location.hash = targetId;
      });
    });

    if (window.location.hash) {
      const hashId = window.location.hash.replace('#', '');
      if (document.getElementById(hashId)) {
        switchTab(hashId);
      }
    }
  }

  function switchTab(targetId) {
    elements.navTabs.forEach(t => {
      const match = t.dataset.target === targetId;
      t.classList.toggle('active', match);
      t.setAttribute('aria-selected', match ? 'true' : 'false');
    });
    elements.tabPanes.forEach(p => {
      p.classList.toggle('active', p.id === targetId);
    });
    state.activeTab = targetId;
  }

  /* -------------------------------------------------------------
     2. PASSENGER PREDICTOR CONTROLS & LOGIC
     ------------------------------------------------------------- */
  function setupPredictorControls() {
    // Model Select
    elements.modelSelect.addEventListener('change', (e) => {
      state.currentModel = e.target.value;
      updateActiveModelHeader();
      triggerSinglePrediction();
      loadManifestDeathCountStats();
    });

    // Class Buttons
    elements.pclassSelector.querySelectorAll('.class-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        elements.pclassSelector.querySelectorAll('.class-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const pclassVal = btn.dataset.class;
        elements.pclassInput.value = pclassVal;
        
        // Auto-adjust realistic default fare if on custom
        if (elements.archetypeSelect.value === 'custom') {
          if (pclassVal === '1') setFare(84);
          else if (pclassVal === '2') setFare(21);
          else setFare(8.05);
        }
        triggerSinglePrediction();
      });
    });

    // Gender Buttons
    elements.genderSelector.querySelectorAll('.gender-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        elements.genderSelector.querySelectorAll('.gender-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        elements.sexInput.value = btn.dataset.sex;
        triggerSinglePrediction();
      });
    });

    // Age Slider & Number Sync
    elements.ageSlider.addEventListener('input', (e) => {
      elements.ageNumber.value = e.target.value;
      elements.ageDisplay.textContent = e.target.value;
      triggerSinglePrediction();
    });

    elements.ageNumber.addEventListener('input', (e) => {
      let val = Math.max(1, Math.min(80, parseInt(e.target.value) || 25));
      elements.ageSlider.value = val;
      elements.ageDisplay.textContent = val;
      triggerSinglePrediction();
    });

    // Family Steppers (SibSp & Parch)
    document.querySelectorAll('.step-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetInput = document.getElementById(btn.dataset.target);
        const delta = parseInt(btn.dataset.delta);
        const currentVal = parseInt(targetInput.value) || 0;
        const min = parseInt(targetInput.min) || 0;
        const max = parseInt(targetInput.max) || 10;
        const nextVal = Math.max(min, Math.min(max, currentVal + delta));
        targetInput.value = nextVal;
        triggerSinglePrediction();
      });
    });

    // Fare Slider
    elements.fareSlider.addEventListener('input', (e) => {
      setFare(parseFloat(e.target.value));
      triggerSinglePrediction();
    });

    // Embarked Select
    elements.embarkedSelect.addEventListener('change', () => {
      triggerSinglePrediction();
    });

    // Predict Button
    elements.predictBtn.addEventListener('click', triggerSinglePrediction);

    // Archetype Presets Dropdown
    elements.archetypeSelect.addEventListener('change', (e) => {
      applyArchetype(e.target.value);
    });
  }

  function setFare(val) {
    elements.fareSlider.value = val;
    elements.fareInput.value = val.toFixed(2);
    elements.fareDisplay.textContent = val.toFixed(2);
  }

  // Passenger names corresponding to presets
  const archetypeNames = {
    custom: 'Custom Passenger',
    jack: 'Mr. Jack Dawson',
    rose: 'Miss Rose DeWitt Bukater',
    captain: 'Capt. Edward J. Smith',
    family_child: 'Master Steerage Child & Mother',
    second_class: 'Mrs. Edwina Troutt (2nd Class)',
    millionaire: 'Col. John Jacob Astor IV'
  };

  function applyArchetype(type) {
    switch (type) {
      case 'jack':
        setPclass(3);
        setGender('male');
        setAge(20);
        setSibsp(0);
        setParch(0);
        setFare(7.25);
        elements.embarkedSelect.value = 'S';
        break;
      case 'rose':
        setPclass(1);
        setGender('female');
        setAge(17);
        setSibsp(0);
        setParch(1);
        setFare(150.0);
        elements.embarkedSelect.value = 'C';
        break;
      case 'captain':
        setPclass(1);
        setGender('male');
        setAge(62);
        setSibsp(0);
        setParch(0);
        setFare(50.0);
        elements.embarkedSelect.value = 'S';
        break;
      case 'family_child':
        setPclass(3);
        setGender('female');
        setAge(8);
        setSibsp(1);
        setParch(1);
        setFare(15.5);
        elements.embarkedSelect.value = 'Q';
        break;
      case 'second_class':
        setPclass(2);
        setGender('female');
        setAge(28);
        setSibsp(0);
        setParch(0);
        setFare(26.0);
        elements.embarkedSelect.value = 'S';
        break;
      case 'millionaire':
        setPclass(1);
        setGender('male');
        setAge(47);
        setSibsp(1);
        setParch(0);
        setFare(247.5);
        elements.embarkedSelect.value = 'C';
        break;
    }
    triggerSinglePrediction();
  }

  function setPclass(pclass) {
    elements.pclassInput.value = pclass;
    elements.pclassSelector.querySelectorAll('.class-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.class == pclass);
    });
  }

  function setGender(sex) {
    elements.sexInput.value = sex;
    elements.genderSelector.querySelectorAll('.gender-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.sex == sex);
    });
  }

  function setAge(age) {
    elements.ageSlider.value = age;
    elements.ageNumber.value = age;
    elements.ageDisplay.textContent = age;
  }

  function setSibsp(n) { elements.sibspInput.value = n; }
  function setParch(n) { elements.parchInput.value = n; }

  /* -------------------------------------------------------------
     3. PREDICTION API CALL & RENDERING
     ------------------------------------------------------------- */
  async function triggerSinglePrediction() {
    const payload = {
      model_key: state.currentModel,
      pclass: parseInt(elements.pclassInput.value),
      sex: elements.sexInput.value,
      age: parseFloat(elements.ageNumber.value),
      sibsp: parseInt(elements.sibspInput.value),
      parch: parseInt(elements.parchInput.value),
      fare: parseFloat(elements.fareInput.value),
      embarked: elements.embarkedSelect.value
    };

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('Prediction API error');
      const data = await res.json();
      renderPredictionResult(data);
    } catch (err) {
      console.error('Error fetching prediction:', err);
    }
  }

  function renderPredictionResult(data) {
    const isSurvived = data.prediction === 1;
    const survivePct = data.survival_probability;
    const deathPct = data.death_probability;

    // 1. Update Radial Gauge
    const totalCirc = 427.26;
    const strokeOffset = totalCirc - (totalCirc * (survivePct / 100));
    elements.gaugeProgress.style.strokeDashoffset = strokeOffset;
    elements.gaugeProgress.style.stroke = isSurvived ? 'var(--color-survived)' : 'var(--color-perished)';
    elements.survivalPctDisplay.textContent = `${survivePct}%`;

    // 2. Status Badge & Outcome Card
    if (isSurvived) {
      elements.statusPulse.className = 'status-indicator-badge survived';
      elements.statusText.textContent = 'PROBABLE SURVIVOR';
      
      elements.mainOutcomeBadge.className = 'outcome-badge-large survived';
      elements.outcomeIcon.className = 'fa-solid fa-life-ring';
      elements.outcomeLabel.textContent = 'SURVIVED';
    } else {
      elements.statusPulse.className = 'status-indicator-badge perished';
      elements.statusText.textContent = 'PROBABLE CASUALTY';
      
      elements.mainOutcomeBadge.className = 'outcome-badge-large perished';
      elements.outcomeIcon.className = 'fa-solid fa-skull';
      elements.outcomeLabel.textContent = 'PERISHED';
    }

    elements.deathPctDisplay.textContent = `${deathPct}%`;
    elements.confidenceDisplay.textContent = `${data.confidence}%`;

    // 3. Ultra-Realistic White Star Line Contract Ticket
    const selectedPreset = elements.archetypeSelect.value;
    const pName = archetypeNames[selectedPreset] || (data.passenger_summary.sex === 'female' ? 'Mrs./Miss Passenger' : 'Mr. Passenger');
    elements.ticketPassengerName.textContent = pName;
    
    elements.ticketNumberDisplay.textContent = `CONTRACT ${data.contract_number}`;
    elements.ticketStubSerial.textContent = `№ ${data.contract_number.replace('No. ', '')}`;
    
    const classNames = { 1: '1ST CLASS LUXURY', 2: '2ND CLASS TOURIST', 3: '3RD CLASS STEERAGE' };
    elements.tClassDisplay.textContent = classNames[data.passenger_summary.pclass] || 'STEERAGE';
    elements.ticketBerthDisplay.textContent = data.assigned_berth || 'Berth Assigned at Southampton';
    
    elements.tPortDisplay.textContent = `${data.passenger_summary.embarked} (10 Apr 1912)`;
    
    const familyCount = data.passenger_summary.sibsp + data.passenger_summary.parch;
    const familyStr = familyCount === 0 ? 'Solo' : `+${familyCount} Companions`;
    elements.tDemographicsDisplay.textContent = `${Math.round(data.passenger_summary.age)} Yrs • ${data.passenger_summary.sex.toUpperCase()} (${familyStr})`;
    
    elements.tFareDisplay.textContent = `£${data.passenger_summary.fare.toFixed(2)} (${data.vintage_fare || ''})`;

    if (isSurvived) {
      elements.ticketStamp.className = 'distressed-rubber-stamp survived';
      elements.ticketStampText.textContent = '★ SURVIVED — RESCUED ★';
      elements.ticketStampSub.textContent = 'PICKED UP BY R.M.S. CARPATHIA';
      elements.ticketStampDate.textContent = '18 APR 1912 • LANDED PORT OF NEW YORK';
    } else {
      elements.ticketStamp.className = 'distressed-rubber-stamp perished';
      elements.ticketStampText.textContent = '★ CASUALTY — PERISHED ★';
      elements.ticketStampSub.textContent = 'S.S. TITANIC FOUNDERED';
      elements.ticketStampDate.textContent = '15th APRIL 1912 • LAT 41°46\'N, LON 50°14\'W';
    }

    // 4. Contributing Factors
    elements.factorsList.innerHTML = '';
    data.risk_factors.forEach(f => {
      const card = document.createElement('div');
      card.className = `factor-card ${f.impact}`;
      card.innerHTML = `
        <div class="factor-title-row">
          <span class="factor-name">${f.factor}</span>
          <span class="factor-strength ${f.impact}">${f.strength} Impact</span>
        </div>
        <p class="factor-text">${f.text}</p>
      `;
      elements.factorsList.appendChild(card);
    });
  }

  /* -------------------------------------------------------------
     4. MANIFEST DEATH COUNT SIMULATOR
     ------------------------------------------------------------- */
  async function loadManifestDeathCountStats() {
    const payload = {
      model_key: state.currentModel,
      pclass_filter: elements.manifestClassFilter.value,
      sex_filter: elements.manifestSexFilter.value,
      embarked_filter: 'all'
    };

    try {
      const res = await fetch('/api/death-count/manifest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('Manifest death count error');
      const data = await res.json();
      renderManifestDeathStats(data);
    } catch (err) {
      console.error('Error loading manifest stats:', err);
    }
  }

  function renderManifestDeathStats(data) {
    if (data.error) {
      alert(data.error);
      return;
    }

    elements.cntTotalPassengers.textContent = data.total_passengers;
    elements.cntPredictedDeaths.textContent = data.predicted_deaths;
    elements.cntActualDeathsSub.textContent = `Actual Historical: ${data.actual_deaths} (${data.actual_mortality_rate}%)`;
    elements.cntPredictedSurvivors.textContent = data.predicted_survivors;
    elements.cntActualSurvivorsSub.textContent = `Actual Historical: ${data.actual_survivors}`;
    elements.cntModelAccuracy.textContent = `${data.death_count_accuracy}%`;
    elements.cntDeathErrorSub.textContent = `Death Toll Delta: ±${data.death_count_error}`;

    // Render Class Breakdown
    elements.classBreakdownList.innerHTML = '';
    for (const [clsName, stats] of Object.entries(data.class_breakdown)) {
      const row = document.createElement('div');
      row.className = 'breakdown-row';
      row.innerHTML = `
        <div class="breakdown-row-header">
          <span class="breakdown-name">${clsName} (n=${stats.total})</span>
          <span class="breakdown-stat-num">${stats.predicted_deaths} Deaths (${stats.predicted_death_rate}%)</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill-perished" style="width: ${stats.predicted_death_rate}%"></div>
        </div>
      `;
      elements.classBreakdownList.appendChild(row);
    }

    // Render Gender Breakdown
    elements.genderBreakdownList.innerHTML = '';
    for (const [genderName, stats] of Object.entries(data.gender_breakdown)) {
      const row = document.createElement('div');
      row.className = 'breakdown-row';
      row.innerHTML = `
        <div class="breakdown-row-header">
          <span class="breakdown-name">${genderName} (n=${stats.total})</span>
          <span class="breakdown-stat-num">${stats.predicted_deaths} Deaths (${stats.predicted_death_rate}%)</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill-perished" style="width: ${stats.predicted_death_rate}%"></div>
        </div>
      `;
      elements.genderBreakdownList.appendChild(row);
    }

    // Render Age Breakdown
    elements.ageBreakdownList.innerHTML = '';
    for (const [ageCat, stats] of Object.entries(data.age_breakdown)) {
      const row = document.createElement('div');
      row.className = 'breakdown-row';
      row.innerHTML = `
        <div class="breakdown-row-header">
          <span class="breakdown-name">${ageCat} (n=${stats.total})</span>
          <span class="breakdown-stat-num">${stats.predicted_deaths} Deaths (${stats.predicted_death_rate}%)</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill-perished" style="width: ${stats.predicted_death_rate}%"></div>
        </div>
      `;
      elements.ageBreakdownList.appendChild(row);
    }
  }

  elements.manifestClassFilter.addEventListener('change', loadManifestDeathCountStats);
  elements.manifestSexFilter.addEventListener('change', loadManifestDeathCountStats);
  elements.refreshManifestStatsBtn.addEventListener('click', loadManifestDeathCountStats);

  /* -------------------------------------------------------------
     5. CUSTOM GROUP SIMULATOR
     ------------------------------------------------------------- */
  function setupCustomSimUI() {
    renderCohorts();

    elements.addCohortBtn.addEventListener('click', () => {
      const newId = Date.now();
      state.customCohorts.push({
        id: newId,
        label: `Cohort #${state.customCohorts.length + 1}`,
        pclass: 3,
        sex: 'male',
        age: 28,
        count: 50,
        fare: 12,
        sibsp: 0,
        parch: 0,
        embarked: 'S'
      });
      renderCohorts();
    });

    elements.runCustomSimBtn.addEventListener('click', runCustomGroupSim);
  }

  function renderCohorts() {
    elements.cohortsContainer.innerHTML = '';
    state.customCohorts.forEach((c, idx) => {
      const row = document.createElement('div');
      row.className = 'cohort-row';
      row.innerHTML = `
        <input type="text" class="glass-input cohort-label" value="${c.label}" placeholder="Label">
        <select class="glass-select cohort-pclass">
          <option value="1" ${c.pclass == 1 ? 'selected' : ''}>1st Class</option>
          <option value="2" ${c.pclass == 2 ? 'selected' : ''}>2nd Class</option>
          <option value="3" ${c.pclass == 3 ? 'selected' : ''}>3rd Class</option>
        </select>
        <select class="glass-select cohort-sex">
          <option value="male" ${c.sex == 'male' ? 'selected' : ''}>Male</option>
          <option value="female" ${c.sex == 'female' ? 'selected' : ''}>Female</option>
        </select>
        <input type="number" class="glass-input cohort-count" value="${c.count}" min="1" max="1000" placeholder="Count">
        <button class="cohort-remove-btn" title="Remove Cohort"><i class="fa-solid fa-trash"></i></button>
      `;

      // Event listeners
      row.querySelector('.cohort-label').addEventListener('change', (e) => c.label = e.target.value);
      row.querySelector('.cohort-pclass').addEventListener('change', (e) => c.pclass = parseInt(e.target.value));
      row.querySelector('.cohort-sex').addEventListener('change', (e) => c.sex = e.target.value);
      row.querySelector('.cohort-count').addEventListener('change', (e) => c.count = parseInt(e.target.value) || 1);
      row.querySelector('.cohort-remove-btn').addEventListener('click', () => {
        state.customCohorts.splice(idx, 1);
        renderCohorts();
      });

      elements.cohortsContainer.appendChild(row);
    });
  }

  async function runCustomGroupSim() {
    if (state.customCohorts.length === 0) {
      alert('Please add at least one cohort to simulate.');
      return;
    }

    try {
      const res = await fetch('/api/death-count/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_key: state.currentModel,
          cohorts: state.customCohorts
        })
      });
      const data = await res.json();
      
      elements.simResultSummary.style.display = 'grid';
      elements.simTotalPeople.textContent = data.total_passengers_simulated;
      elements.simTotalDeaths.textContent = data.total_predicted_deaths;
      elements.simTotalSurvivors.textContent = data.total_predicted_survivors;
      elements.simMortalityRate.textContent = `${data.overall_mortality_rate}%`;
    } catch (err) {
      console.error('Error simulating custom group:', err);
    }
  }

  /* -------------------------------------------------------------
     6. CSV UPLOAD BATCH PREDICTOR
     ------------------------------------------------------------- */
  function setupCSVUpload() {
    const dropzone = elements.csvDropzone;
    const fileInput = elements.csvFileInput;

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length > 0) {
        handleCSVFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleCSVFile(e.target.files[0]);
      }
    });
  }

  function handleCSVFile(file) {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const csvText = e.target.result;
      try {
        const res = await fetch('/api/upload-csv', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: file.name,
            model_key: state.currentModel,
            csv_text: csvText
          })
        });
        if (!res.ok) {
          const err = await res.json();
          alert(err.detail || 'Failed to process CSV');
          return;
        }
        const data = await res.json();
        renderCSVResults(data);
      } catch (err) {
        console.error('Error uploading CSV:', err);
      }
    };
    reader.readAsText(file);
  }

  function renderCSVResults(data) {
    elements.csvResultBox.style.display = 'flex';
    elements.csvFileName.innerHTML = `<i class="fa-solid fa-file-csv"></i> ${data.filename} (${data.total_passengers} passengers)`;
    elements.csvDeaths.textContent = `${data.predicted_deaths} (${data.mortality_rate_pct}%)`;
    elements.csvSurvivors.textContent = `${data.predicted_survivors}`;

    elements.csvTableBody.innerHTML = '';
    data.predictions.forEach(p => {
      const tr = document.createElement('tr');
      const isSurv = p.prediction === 1;
      tr.innerHTML = `
        <td>${p.row}</td>
        <td>Class ${p.pclass}</td>
        <td>${p.sex}</td>
        <td>${Math.round(p.age)}</td>
        <td>£${p.fare.toFixed(2)}</td>
        <td><span class="badge-tag ${isSurv ? 'survived' : 'perished'}">${p.outcome}</span></td>
        <td>${p.death_probability}%</td>
      `;
      elements.csvTableBody.appendChild(tr);
    });
  }

  /* -------------------------------------------------------------
     7. MODEL BENCHMARK LAB DATA
     ------------------------------------------------------------- */
  async function loadModelsData() {
    try {
      const res = await fetch('/api/models');
      const models = await res.json();
      state.models = models;
      renderModelCards(models);
      renderConfusionMatrices(models);
      updateActiveModelHeader();
    } catch (err) {
      console.error('Error loading models data:', err);
    }
  }

  function updateActiveModelHeader() {
    const m = state.models[state.currentModel];
    if (m) {
      elements.activeModelHeader.textContent = `${m.name.split('(')[0]} • ${m.accuracy}% Acc`;
    }
  }

  function renderModelCards(models) {
    elements.modelsBenchmarkGrid.innerHTML = '';
    for (const [key, m] of Object.entries(models)) {
      const card = document.createElement('div');
      card.className = `model-card ${m.recommended ? 'recommended' : ''}`;
      card.innerHTML = `
        <div class="model-card-top-bar">
          <span class="model-type-badge"><i class="fa-solid fa-microchip"></i> CLASSIFIER</span>
          ${m.recommended ? '<span class="model-ribbon"><i class="fa-solid fa-star"></i> TOP PICK</span>' : ''}
        </div>
        <div class="model-card-header">
          <h3 class="model-card-title">${m.name}</h3>
          <p class="model-desc">${m.description}</p>
        </div>
        <div class="model-metrics-row">
          <div class="metric-item">
            <span class="m-label">ACCURACY</span>
            <span class="m-val acc">${m.accuracy}%</span>
          </div>
          <div class="metric-item">
            <span class="m-label">PRECISION</span>
            <span class="m-val">${m.precision}%</span>
          </div>
          <div class="metric-item">
            <span class="m-label">F1-SCORE</span>
            <span class="m-val">${m.f1_score}%</span>
          </div>
        </div>
        <div class="manifest-death-stat-badge">
          <span>Full Manifest Deaths Predicted: <strong>${m.manifest_stats.predicted_deaths}</strong> (Actual: 549)</span>
        </div>
      `;
      elements.modelsBenchmarkGrid.appendChild(card);
    }
  }

  function renderConfusionMatrices(models) {
    elements.cmDisplayGrid.innerHTML = '';
    for (const [key, m] of Object.entries(models)) {
      const cm = m.confusion_matrix;
      if (!cm) continue;
      const box = document.createElement('div');
      box.className = 'cm-box';
      box.innerHTML = `
        <h4>${m.name}</h4>
        <table class="cm-matrix-table">
          <thead>
            <tr>
              <th></th>
              <th>Pred: Died</th>
              <th>Pred: Survived</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>Act: Died</th>
              <td class="tn">${cm[0][0]} (TN)</td>
              <td class="fp">${cm[0][1]} (FP)</td>
            </tr>
            <tr>
              <th>Act: Survived</th>
              <td class="fn">${cm[1][0]} (FN)</td>
              <td class="tp">${cm[1][1]} (TP)</td>
            </tr>
          </tbody>
        </table>
      `;
      elements.cmDisplayGrid.appendChild(box);
    }
  }

  /* -------------------------------------------------------------
     8. HISTORICAL MANIFEST TABLE & SEARCH
     ------------------------------------------------------------- */
  function setupManifestSearch() {
    let timeout;
    elements.manifestSearchInput.addEventListener('input', (e) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        state.manifestSearch = e.target.value.trim();
        state.manifestPage = 0;
        loadHistoricalManifest();
      }, 300);
    });

    elements.prevManifestPage.addEventListener('click', () => {
      if (state.manifestPage > 0) {
        state.manifestPage--;
        loadHistoricalManifest();
      }
    });

    elements.nextManifestPage.addEventListener('click', () => {
      state.manifestPage++;
      loadHistoricalManifest();
    });
  }

  async function loadHistoricalManifest() {
    const offset = state.manifestPage * state.manifestLimit;
    const url = `/api/historical-manifest?limit=${state.manifestLimit}&offset=${offset}&search=${encodeURIComponent(state.manifestSearch)}`;

    try {
      const res = await fetch(url);
      const data = await res.json();
      
      elements.historicalTableBody.innerHTML = '';
      data.records.forEach(r => {
        const isSurv = r.survived === 1;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>#${r.passenger_id}</td>
          <td>Class ${r.pclass}</td>
          <td>${r.sex}</td>
          <td>${r.who}</td>
          <td>${r.age !== null ? Math.round(r.age) : '—'}</td>
          <td>${r.sibsp + r.parch}</td>
          <td>£${r.fare.toFixed(2)}</td>
          <td>${r.embark_town}</td>
          <td><span class="badge-tag ${isSurv ? 'survived' : 'perished'}">${isSurv ? 'SURVIVED' : 'PERISHED'}</span></td>
          <td><span class="badge-tag ${isSurv ? 'survived' : 'perished'}">${isSurv ? 'SURVIVED' : 'PERISHED'}</span></td>
        `;
        elements.historicalTableBody.appendChild(tr);
      });

      const start = offset + 1;
      const end = Math.min(offset + state.manifestLimit, data.total);
      elements.manifestPageIndicator.textContent = `Showing ${start} - ${end} of ${data.total}`;
      elements.prevManifestPage.disabled = state.manifestPage === 0;
      elements.nextManifestPage.disabled = end >= data.total;
    } catch (err) {
      console.error('Error loading historical manifest:', err);
    }
  }

});
