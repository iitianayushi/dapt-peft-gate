function compareUtf8(a, b) {
  return Buffer.from(String(a), 'utf8').compare(Buffer.from(String(b), 'utf8'));
}

function isSafeInteger(val) {
  return typeof val === 'number' && Number.isSafeInteger(val) && !isNaN(val);
}

function isPositiveSafeInt(val) {
  return isSafeInteger(val) && val > 0;
}

function isNonNegativeSafeInt(val) {
  return isSafeInteger(val) && val >= 0;
}

function isUnitInterval(val) {
  return typeof val === 'number' && Number.isFinite(val) && val >= 0 && val <= 1;
}

function isNonNegativeFinite(val) {
  return typeof val === 'number' && Number.isFinite(val) && val >= 0;
}

function round12(num) {
  return Number(Math.round(Number(num + 'e12')) + 'e-12');
}

const PUBLISHED_INTERVENTIONS = ['prompt_only', 'retrieval', 'lora', 'qlora'];
const HEX_40_REGEX = /^[0-9a-f]{40}$/;
const HEX_64_REGEX = /^[0-9a-f]{64}$/;

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'METHOD_NOT_ALLOWED' });
  }

  const body = req.body;
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return res.status(400).json({ error: 'INVALID_INPUT' });
  }

  const { operation } = body;
  if (operation !== 'choose' && operation !== 'repair') {
    return res.status(400).json({ error: 'INVALID_INPUT' });
  }

  // =========================================================================
  // OPERATION 1: CHOOSE
  // =========================================================================
  if (operation === 'choose') {
    const { policy, candidates } = body;

    let isPolicyValid = (
      policy && typeof policy === 'object' && !Array.isArray(policy) &&
      isUnitInterval(policy.minQuality) &&
      typeof policy.freshnessRequired === 'boolean' &&
      isNonNegativeFinite(policy.maxLatencyMs) &&
      isNonNegativeFinite(policy.maxMemoryMb) &&
      isNonNegativeSafeInt(policy.maxLabeledExamples) &&
      isNonNegativeFinite(policy.maxTotalCost) &&
      isNonNegativeSafeInt(policy.horizonRequests)
    );

    const totalCosts = {};
    const reasonCodes = {};
    for (const name of PUBLISHED_INTERVENTIONS) {
      totalCosts[name] = null;
      reasonCodes[name] = [];
    }

    if (!isPolicyValid || !Array.isArray(candidates) || candidates.length !== 4) {
      for (const name of PUBLISHED_INTERVENTIONS) {
        reasonCodes[name] = ['INVALID_INPUT'];
      }
      return res.status(200).json({
        selected: null,
        eligible: [],
        totalCosts,
        reasonCodes
      });
    }

    const candidateMap = new Map();
    let hasDuplicateOrUnknown = false;

    for (const c of candidates) {
      if (!c || typeof c !== 'object' || Array.isArray(c) || typeof c.name !== 'string') {
        hasDuplicateOrUnknown = true;
        break;
      }
      if (!PUBLISHED_INTERVENTIONS.includes(c.name) || candidateMap.has(c.name)) {
        hasDuplicateOrUnknown = true;
        break;
      }
      candidateMap.set(c.name, c);
    }

    if (hasDuplicateOrUnknown || candidateMap.size !== 4) {
      for (const name of PUBLISHED_INTERVENTIONS) {
        reasonCodes[name] = ['INVALID_INPUT'];
      }
      return res.status(200).json({
        selected: null,
        eligible: [],
        totalCosts,
        reasonCodes
      });
    }

    const eligible = [];

    for (const name of PUBLISHED_INTERVENTIONS) {
      const c = candidateMap.get(name);
      const codes = new Set();

      const isValidShape = (
        typeof c.available === 'boolean' &&
        isUnitInterval(c.quality) &&
        typeof c.freshness === 'boolean' &&
        isNonNegativeFinite(c.latencyMs) &&
        isNonNegativeFinite(c.memoryMb) &&
        isNonNegativeSafeInt(c.labeledExamples) &&
        isNonNegativeFinite(c.oneTimeCost) &&
        isNonNegativeFinite(c.recurringCost)
      );

      if (!isValidShape) {
        codes.add('INVALID_INPUT');
      } else {
        const computedCost = round12(c.oneTimeCost + policy.horizonRequests * c.recurringCost);
        totalCosts[name] = computedCost;

        if (!c.available) codes.add('UNAVAILABLE');
        if (c.quality < policy.minQuality) codes.add('QUALITY_FLOOR');
        if (policy.freshnessRequired && !c.freshness) codes.add('FRESHNESS_REQUIRED');
        if (c.latencyMs > policy.maxLatencyMs) codes.add('LATENCY_LIMIT');
        if (c.memoryMb > policy.maxMemoryMb) codes.add('MEMORY_LIMIT');
        if (c.labeledExamples > policy.maxLabeledExamples) codes.add('DATA_LIMIT');
        if (computedCost > policy.maxTotalCost) codes.add('COST_LIMIT');
      }

      const sortedCodes = Array.from(codes).sort(compareUtf8);
      reasonCodes[name] = sortedCodes;

      if (sortedCodes.length === 0) {
        eligible.push(name);
      }
    }

    const selected = eligible.length > 0 ? eligible[0] : null;

    return res.status(200).json({
      selected,
      eligible,
      totalCosts,
      reasonCodes
    });
  }

  // =========================================================================
  // OPERATION 2: REPAIR
  // =========================================================================
  if (operation === 'repair') {
    const {
      tokens,
      templateApplications,
      parameters,
      allowedTargets,
      inferenceMode,
      trainRowIds,
      evalRowIds,
      dropoutActiveDuringEval,
      artifactFiles,
      baseRevision,
      datasetDigest,
      codeDigest,
      configDigest,
      expectedDigests,
      microBatch,
      gradientAccumulation,
      replicas,
      expectedEffectiveBatch,
      checkpoint,
      uninterruptedWeights,
      resumedWeights,
      resumeTolerance
    } = body;

    const reasonCodes = new Set();

    // 1. Tokens & Loss Masking
    let tokensValid = Array.isArray(tokens) && tokens.length > 0;
    if (tokensValid) {
      for (const t of tokens) {
        if (
          !t || typeof t !== 'object' || Array.isArray(t) ||
          !isNonNegativeSafeInt(t.id) ||
          (t.role !== 'system' && t.role !== 'user' && t.role !== 'assistant') ||
          typeof t.padding !== 'boolean' ||
          typeof t.text !== 'string'
        ) {
          tokensValid = false;
          reasonCodes.add('INVALID_TOKEN');
          break;
        }
      }
    } else {
      reasonCodes.add('INVALID_TOKEN');
    }

    let labels = [];
    if (!tokensValid) {
      const len = Array.isArray(tokens) ? tokens.length : 0;
      labels = new Array(len).fill(-100);
    } else {
      labels = tokens.map(t => (t.role === 'assistant' && !t.padding ? t.id : -100));
    }

    // 2. Chat Template
    const templatePass = templateApplications === 1;
    if (!templatePass) {
      reasonCodes.add('CHAT_TEMPLATE_COUNT');
    }

    // 3. Parameters & PEFT Config
    let paramsValid = Array.isArray(parameters) && Array.isArray(allowedTargets);
    const seenParamNames = new Set();
    const allowedTargetSet = new Set();

    if (paramsValid) {
      for (const target of allowedTargets) {
        if (typeof target !== 'string' || target.length === 0 || allowedTargetSet.has(target)) {
          paramsValid = false;
          break;
        }
        allowedTargetSet.add(target);
      }
      if (allowedTargetSet.size === 0) paramsValid = false;
    }

    const trainableParamNames = [];
    let trainableCount = 0;
    let hasAllowedLoraParam = false;

    if (paramsValid) {
      for (const p of parameters) {
        if (
          !p || typeof p !== 'object' || Array.isArray(p) ||
          typeof p.name !== 'string' || p.name.length === 0 || seenParamNames.has(p.name) ||
          typeof p.target !== 'string' ||
          !isPositiveSafeInt(p.numel)
        ) {
          paramsValid = false;
          break;
        }
        seenParamNames.add(p.name);

        const isLora = p.name.endsWith('.lora_A.weight') || p.name.endsWith('.lora_B.weight');
        const isAllowedTarget = allowedTargetSet.has(p.target);

        if (isLora && isAllowedTarget) {
          hasAllowedLoraParam = true;
          trainableParamNames.push(p.name);
          trainableCount += p.numel;
        }
      }
    }

    if (!paramsValid || !hasAllowedLoraParam) {
      reasonCodes.add('INVALID_PARAMETER');
    }

    trainableParamNames.sort(compareUtf8);

    if (inferenceMode !== false) {
      reasonCodes.add('INFERENCE_MODE');
    }
    const peftConfigPass = paramsValid && hasAllowedLoraParam && inferenceMode === false;

    // 4. Artifact Files
    let adapterFiles = [];
    if (Array.isArray(artifactFiles)) {
      const distinctArtifacts = new Set(artifactFiles);
      if (
        artifactFiles.length === 2 &&
        distinctArtifacts.size === 2 &&
        distinctArtifacts.has('adapter_config.json') &&
        distinctArtifacts.has('adapter_model.safetensors')
      ) {
        adapterFiles = ['adapter_config.json', 'adapter_model.safetensors'].sort(compareUtf8);
      } else {
        reasonCodes.add('ADAPTER_FILE_SET');
        if (artifactFiles.some(f => typeof f === 'string' && (f.includes('pytorch_model') || f.includes('model.safetensors')))) {
          reasonCodes.add('FULL_MODEL_ARTIFACT');
        }
      }
    } else {
      reasonCodes.add('ADAPTER_FILE_SET');
    }

    // 5. Checkpoint Completeness
    const requiredCkptKeys = ['model', 'optimizer', 'scheduler', 'step', 'rng', 'dataPosition'];
    let checkpointComplete = checkpoint !== null && typeof checkpoint === 'object' && !Array.isArray(checkpoint);
    if (checkpointComplete) {
      for (const k of requiredCkptKeys) {
        if (!(k in checkpoint)) {
          checkpointComplete = false;
          break;
        }
      }
    }
    if (!checkpointComplete) {
      reasonCodes.add('INCOMPLETE_CHECKPOINT');
    }

    // 6. Lineage & Base Revision
    let lineagePass = true;
    if (typeof baseRevision !== 'string' || !HEX_40_REGEX.test(baseRevision)) {
      lineagePass = false;
      reasonCodes.add('MUTABLE_BASE_REVISION');
    }

    const digestsValid = (
      typeof datasetDigest === 'string' && HEX_64_REGEX.test(datasetDigest) &&
      typeof codeDigest === 'string' && HEX_64_REGEX.test(codeDigest) &&
      typeof configDigest === 'string' && HEX_64_REGEX.test(configDigest) &&
      expectedDigests && typeof expectedDigests === 'object' && !Array.isArray(expectedDigests) &&
      expectedDigests.datasetDigest === datasetDigest &&
      expectedDigests.codeDigest === codeDigest &&
      expectedDigests.configDigest === configDigest
    );

    if (!digestsValid) {
      lineagePass = false;
      reasonCodes.add('LINEAGE_MISMATCH');
    }

    // 7. Effective Batch
    const batchFactorsValid = (
      isPositiveSafeInt(microBatch) &&
      isPositiveSafeInt(gradientAccumulation) &&
      isPositiveSafeInt(replicas) &&
      isPositiveSafeInt(expectedEffectiveBatch) &&
      (microBatch * gradientAccumulation * replicas === expectedEffectiveBatch)
    );
    if (!batchFactorsValid) {
      reasonCodes.add('EFFECTIVE_BATCH_MISMATCH');
    }

    // 8. Evaluation Isolation & Dropout
    let evalIsolated = true;
    let evalRowsValid = Array.isArray(trainRowIds) && Array.isArray(evalRowIds) && trainRowIds.length > 0 && evalRowIds.length > 0;
    const trainSet = new Set();
    const evalSet = new Set();

    if (evalRowsValid) {
      for (const id of trainRowIds) {
        if (typeof id !== 'string' || id.length === 0 || trainSet.has(id)) { evalRowsValid = false; break; }
        trainSet.add(id);
      }
      for (const id of evalRowIds) {
        if (typeof id !== 'string' || id.length === 0 || evalSet.has(id)) { evalRowsValid = false; break; }
        evalSet.add(id);
      }
    }

    if (!evalRowsValid) {
      evalIsolated = false;
      reasonCodes.add('EVAL_LEAKAGE');
    } else {
      for (const id of evalSet) {
        if (trainSet.has(id)) {
          evalIsolated = false;
          reasonCodes.add('EVAL_LEAKAGE');
          break;
        }
      }
    }

    const evaluationDeterministic = dropoutActiveDuringEval === false;
    if (!evaluationDeterministic) {
      reasonCodes.add('EVAL_DROPOUT_ACTIVE');
    }

    // 9. Resume Divergence
    let resumePass = true;
    const resumeArraysValid = (
      Array.isArray(uninterruptedWeights) &&
      Array.isArray(resumedWeights) &&
      uninterruptedWeights.length > 0 &&
      uninterruptedWeights.length === resumedWeights.length &&
      isNonNegativeFinite(resumeTolerance) &&
      uninterruptedWeights.every(v => typeof v === 'number' && Number.isFinite(v)) &&
      resumedWeights.every(v => typeof v === 'number' && Number.isFinite(v))
    );

    if (!resumeArraysValid) {
      resumePass = false;
      reasonCodes.add('RESUME_DIVERGENCE');
    } else {
      for (let i = 0; i < uninterruptedWeights.length; i++) {
        const diff = Math.abs(uninterruptedWeights[i] - resumedWeights[i]);
        if (diff > resumeTolerance) {
          resumePass = false;
          reasonCodes.add('RESUME_DIVERGENCE');
          break;
        }
      }
    }

    const finalReasonCodes = Array.from(reasonCodes).sort(compareUtf8);

    return res.status(200).json({
      labels,
      templatePass,
      trainableParams: trainableParamNames,
      trainableCount,
      peftConfigPass,
      adapterFiles,
      checkpointComplete,
      lineagePass,
      evalIsolated,
      evaluationDeterministic,
      resumePass,
      reasonCodes: finalReasonCodes
    });
  }
}
