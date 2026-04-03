// Tour step definitions - each step maps to a quadrant position,
// narration text, and which interactive component to show.

export default function getTourSteps(t) {
  return [
    // === INTRO ===
    {
      id: 'welcome',
      quadrant: null,
      shortTitle: t('tour.welcome.shortTitle'),
      title: t('tour.welcome.title'),
      narration: t('tour.welcome.narration'),
      component: 'Welcome',
    },

    // === PROMPT ENGINEERING (Lower-Left) ===
    {
      id: 'prompt-basic',
      quadrant: 'prompt',
      shortTitle: t('tour.promptBasic.shortTitle'),
      subStop: null,
      zigzagPosition: 0,
      title: t('tour.promptBasic.title'),
      narration: t('tour.promptBasic.narration'),
      component: 'PromptBasic',
    },
    {
      id: 'prompt-fewshot',
      quadrant: 'prompt',
      shortTitle: t('tour.promptFewShot.shortTitle'),
      subStop: null,
      zigzagPosition: 1,
      title: t('tour.promptFewShot.title'),
      narration: t('tour.promptFewShot.narration'),
      component: 'PromptFewShot',
    },
    {
      id: 'prompt-limitation',
      quadrant: 'prompt',
      shortTitle: t('tour.promptLimitation.shortTitle'),
      subStop: null,
      zigzagPosition: 1,
      title: t('tour.promptLimitation.title'),
      narration: t('tour.promptLimitation.narration'),
      component: 'PromptLimitation',
    },

    // === RAG (Upper-Left) ===
    {
      id: 'rag-simple',
      quadrant: 'rag',
      shortTitle: t('tour.ragSimple.shortTitle'),
      subStop: null,
      zigzagPosition: 2,
      title: t('tour.ragSimple.title'),
      narration: t('tour.ragSimple.narration'),
      component: 'RAGSimple',
    },
    {
      id: 'rag-limitation',
      quadrant: 'rag',
      shortTitle: t('tour.ragLimitation.shortTitle'),
      subStop: null,
      zigzagPosition: 3,
      title: t('tour.ragLimitation.title'),
      narration: t('tour.ragLimitation.narration'),
      component: 'RAGLimitation',
    },

    // === POST-TRAINING: SFT (Lower-Right) ===
    {
      id: 'sft',
      quadrant: 'posttraining',
      shortTitle: t('tour.sft.shortTitle'),
      subStop: 'sft',
      zigzagPosition: 4,
      title: t('tour.sft.title'),
      narration: t('tour.sft.narration'),
      component: 'SFTComparison',
    },

    // === POST-TRAINING: DPO (Lower-Right, sub-stop) ===
    {
      id: 'dpo',
      quadrant: 'posttraining',
      shortTitle: t('tour.dpo.shortTitle'),
      subStop: 'dpo',
      zigzagPosition: 5,
      title: t('tour.dpo.title'),
      narration: t('tour.dpo.narration'),
      component: 'DPOPreferences',
    },

    // === POST-TRAINING: GRPO (Lower-Right, sub-stop) ===
    {
      id: 'grpo',
      quadrant: 'posttraining',
      shortTitle: t('tour.grpo.shortTitle'),
      subStop: 'grpo',
      zigzagPosition: 6,
      title: t('tour.grpo.title'),
      narration: t('tour.grpo.narration'),
      component: 'GRPOGenerations',
    },

    // === ALL OPTIONS (Upper-Right) ===
    {
      id: 'combined',
      quadrant: 'alloptions',
      shortTitle: t('tour.combined.shortTitle'),
      subStop: null,
      zigzagPosition: 7,
      title: t('tour.combined.title'),
      narration: t('tour.combined.narration'),
      component: 'CombinedResults',
    },
    {
      id: 'infrastructure',
      quadrant: 'alloptions',
      shortTitle: t('tour.infrastructure.shortTitle'),
      subStop: null,
      zigzagPosition: 7,
      title: t('tour.infrastructure.title'),
      narration: t('tour.infrastructure.narration'),
      component: 'InfrastructureSummary',
    },

    // === EPILOGUE ===
    {
      id: 'epilogue',
      quadrant: null,
      shortTitle: t('tour.epilogue.shortTitle'),
      title: t('tour.epilogue.title'),
      narration: t('tour.epilogue.narration'),
      component: 'Epilogue',
    },
  ]
}
