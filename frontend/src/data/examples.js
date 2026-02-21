export const GOOD_EXAMPLES = [
  {
    id: 'gp1',
    title: 'Quantum Foundations',
    gradient: 'linear-gradient(135deg, #4c1d95 0%, #1e1b4b 100%)',
    text: `Analyze the epistemological foundations of quantum entanglement by integrating formal mathematical structure, experimental validation, and philosophical interpretation into a coherent explanatory framework. Begin by describing how tensor product Hilbert spaces allow composite quantum systems to exhibit non-factorizable state vectors, and clarify why separability fails under entangled configurations. Then examine Bell's inequalities, including the CHSH formulation, and explain how empirical violations observed in Aspect-type experiments undermine classical locality and deterministic realism. Extend the discussion toward decoherence theory, entropic correlations, and the role of measurement operators in collapsing superposed amplitudes. Contrast Copenhagen, Many-Worlds, and relational interpretations, focusing specifically on their ontological commitments and metaphysical implications. Additionally, evaluate how quantum information theory reframes entanglement as a computational resource enabling teleportation, superdense coding, and cryptographic security. Finally, synthesize these perspectives into a structured argument addressing whether entanglement necessitates nonlocal causation or instead demands a revision of classical intuitions regarding separability, causality, and physical realism.`,
  },
  {
    id: 'gp2',
    title: 'Photosynthesis Systems',
    gradient: 'linear-gradient(135deg, #065f46 0%, #064e3b 100%)',
    text: `Construct a systems-level biochemical and thermodynamic analysis of photosynthesis that integrates molecular structure, energetic transfer mechanisms, and ecological macro-dynamics. Begin by formally describing chloroplast ultrastructure and pigment absorption spectra in terms of quantum excitation states. Then analyze the light-dependent reactions as an electron transport optimization problem, including photolysis, proton gradients, chemiosmotic coupling, and ATP synthase rotation mechanics. Extend the discussion into the Calvin-Benson cycle using carbon fixation kinetics, RuBisCO efficiency constraints, and NADPH reduction pathways. Evaluate photosynthesis as an entropy-management system that converts low-entropy solar radiation into high-order biochemical organization. Finally, synthesize its planetary-scale implications for atmospheric regulation, carbon sequestration feedback loops, and biospheric energy flow stability.`,
  },
  {
    id: 'gp3',
    title: 'Printing Press Analysis',
    gradient: 'linear-gradient(135deg, #9f1239 0%, #4c0519 100%)',
    text: `Develop a multi-layered historical and epistemological examination of the Gutenberg printing press by integrating technological innovation theory, sociopolitical restructuring, and cognitive-cultural transformation. Begin by describing the mechanical engineering principles underlying movable type standardization and ink transfer reproducibility. Then analyze how mass replication altered information diffusion velocity and network topology across Renaissance Europe. Evaluate its causal role in accelerating scientific method formalization, destabilizing ecclesiastical epistemic monopolies, and enabling vernacular linguistic codification. Extend the analysis toward media ecology theory and distributed cognition, examining how print culture reshaped memory externalization and authority structures. Conclude by synthesizing how the printing press functioned as an epistemic amplifier that reconfigured knowledge production, institutional legitimacy, and political sovereignty.`,
  },
  {
    id: 'gp4',
    title: 'ML Paradigm Theory',
    gradient: 'linear-gradient(135deg, #1e40af 0%, #172554 100%)',
    text: `Produce a mathematically grounded and architecturally comparative analysis of supervised and unsupervised machine learning paradigms, emphasizing objective functions, representational geometry, and statistical inference principles. Begin by defining supervised learning as an empirical risk minimization framework over labeled distributions and contrast it with unsupervised latent-variable modeling and manifold estimation. Analyze bias-variance trade-offs, generalization bounds, and overfitting dynamics under distributional shift. Compare algorithmic mechanisms such as Support Vector Machines, ensemble-based decision forests, K-Means clustering, and Principal Component Analysis through the lens of optimization landscapes and feature-space transformations. Extend the discussion toward interpretability constraints, scalability limits, and robustness under adversarial perturbations. Finally, synthesize these paradigms into a structured framework evaluating when hybrid semi-supervised or self-supervised approaches become epistemically advantageous.`,
  },
];

export const BAD_EXAMPLES = [
  { id: 'bp1', label: 'AI Basic', text: 'Explain artificial intelligence in simple terms.' },
  { id: 'bp2', label: 'Computers', text: 'Describe how computers work.' },
  { id: 'bp3', label: 'WWII Summary', text: 'Give a summary of World War II.' },
  { id: 'bp4', label: 'Sky Simple', text: 'Explain why the sky is blue in a short answer.' },
];

export const GROQ_MODELS = [
  'llama-3.3-70b-versatile',
  'llama-3.1-8b-instant',
  'qwen/qwen3-32b',
  'groq/compound',
  'groq/compound-mini',
  'openai/gpt-oss-120b',
  'openai/gpt-oss-20b',
];

export const GEMINI_MODELS = [
  'gemini-3-pro-preview',
  'gemini-3-flash-preview',
  'gemini-2.5-pro',
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite',
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite',
  'gemini-flash-latest',
  'gemini-flash-lite-latest',
  'gemini-robotics-er-1.5-preview',
];

export const OPENAI_MODELS = [
  'gpt-5.2',
  'gpt-5-mini',
  'gpt-5-nano',
  'gpt-5.2-pro',
  'gpt-5',
  'gpt-4.1',
  'gpt-4o',
  'gpt-4-turbo',
  'gpt-4',
  'gpt-3.5-turbo',
  'gpt-oss-120b',
  'gpt-oss-20b',
];
