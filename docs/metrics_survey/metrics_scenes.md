# Evaluation metrics in indoor scene synthesis / 3D room layout generation (2019–2026)

Compiled 2026-10-21. I read every definition below from the paper's PDF on arXiv (full text via `arxiv.org/pdf/<id>`, then searched for the evaluation section and appendix). The URL next to each paper is the one I read. Exact quotes are in "double quotes". Where I could not verify a detail, the text says so.

## 0. Papers read (ID → URL, venue)

| Key | Paper | Venue / year | URL read |
|---|---|---|---|
| GRAINS | GRAINS: Generative Recursive Autoencoders for INdoor Scenes | ACM TOG 2019 | https://arxiv.org/abs/1807.09193 |
| SceneFormer | SceneFormer: Indoor Scene Generation with Transformers | 3DV 2021 | https://arxiv.org/abs/2012.09793 |
| ATISS | ATISS: Autoregressive Transformers for Indoor Scene Synthesis | NeurIPS 2021 | https://arxiv.org/abs/2110.03675 |
| ProcTHOR | ProcTHOR: Large-Scale Embodied AI Using Procedural Generation | NeurIPS 2022 | https://arxiv.org/abs/2206.06994 |
| LEGO-Net | LEGO-Net: Learning Regular Rearrangements of Objects in Rooms | CVPR 2023 | https://arxiv.org/abs/2301.09629 |
| Set-the-Scene | Set-the-Scene: Global-Local Training for Controllable NeRF Scenes | ICCV 2023 | https://arxiv.org/abs/2303.13450 |
| LayoutGPT | LayoutGPT: Compositional Visual Planning and Generation with LLMs | NeurIPS 2023 | https://arxiv.org/abs/2305.15393 |
| CommonScenes | CommonScenes: Commonsense 3D Indoor Scenes with Scene Graph Diffusion | NeurIPS 2023 | https://arxiv.org/abs/2305.16283 |
| DiffuScene | DiffuScene: Denoising Diffusion Models for Generative Indoor Scene Synthesis | CVPR 2024 | https://arxiv.org/abs/2303.14207 |
| InstructScene | InstructScene: Instruction-Driven 3D Indoor Scene Synthesis with Semantic Graph Prior | ICLR 2024 | https://arxiv.org/abs/2402.04717 |
| PhyScene | PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI | CVPR 2024 | https://arxiv.org/abs/2404.09465 |
| Holodeck | Holodeck: Language Guided Generation of 3D Embodied AI Environments | CVPR 2024 | https://arxiv.org/abs/2312.09067 |
| Infinigen-Indoors | Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation | CVPR 2024 | https://arxiv.org/abs/2406.11824 |
| Open-Universe | Open-Universe Indoor Scene Generation using LLM Program Synthesis (Aguina-Kang et al.) | arXiv 2024 | https://arxiv.org/abs/2403.09675 |
| FurniScene | FurniScene: A Large-scale 3D Room Dataset with Intricate Furnishing Scenes | arXiv 2024 | https://arxiv.org/abs/2401.03470 |
| SceneCraft-agent | SceneCraft: An LLM Agent for Synthesizing 3D Scene as Blender Code (Hu et al.) | ICML 2024 | https://arxiv.org/abs/2403.01248 |
| I-Design | I-Design: Personalized LLM Interior Designer | ECCV 2024 Workshops | https://arxiv.org/abs/2404.02838 |
| AnyHome | AnyHome: Open-Vocabulary Generation of Structured and Textured 3D Homes | ECCV 2024 | https://arxiv.org/abs/2312.06644 |
| SceneTeller | SceneTeller: Language-to-3D Scene Generation | ECCV 2024 | https://arxiv.org/abs/2407.20727 |
| EchoScene | EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion | ECCV 2024 | https://arxiv.org/abs/2405.00915 |
| MiDiffusion | Mixed Diffusion for 3D Indoor Scene Synthesis | arXiv 2024 | https://arxiv.org/abs/2405.21066 |
| DeBaRA | DeBaRA: Denoising-Based 3D Room Arrangement Generation | NeurIPS 2024 | https://arxiv.org/abs/2409.18336 |
| SceneCraft-layout | SceneCraft: Layout-Guided 3D Scene Generation (Yang et al.) | NeurIPS 2024 | https://arxiv.org/abs/2410.09049 |
| Architect | Architect: Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting | NeurIPS 2024 | https://arxiv.org/abs/2411.09823 |
| LayoutVLM | LayoutVLM: Differentiable Optimization of 3D Layout via VLMs | CVPR 2025 | https://arxiv.org/abs/2412.02193 |
| FlairGPT | FlairGPT: Repurposing LLMs for Interior Designs | CGF / Eurographics 2025 | https://arxiv.org/abs/2501.04648 |
| FirePlace | FirePlace: Geometric Refinements of LLM Common Sense Reasoning for 3D Object Placement | CVPR 2025 | https://arxiv.org/abs/2503.04919 |
| GLTS | Global-Local Tree Search in VLMs for 3D Indoor Scene Generation | CVPR 2025 | https://arxiv.org/abs/2503.18476 |
| SceneEval | SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis | WACV 2026 (arXiv 2025) | https://arxiv.org/abs/2503.14756 |
| HSM | HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation | arXiv 2025 (v3 Dec 2025) | https://arxiv.org/abs/2503.16848 |
| Scenethesis | Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation | ICLR 2026 | https://arxiv.org/abs/2505.02836 |
| ReSpace | ReSpace: Text-Driven Autoregressive 3D Indoor Scene Synthesis and Editing | arXiv 2025 | https://arxiv.org/abs/2506.02459 |
| DirectLayout | Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning | arXiv 2025 | https://arxiv.org/abs/2506.05341 |
| OptiScene | LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization | arXiv 2025 | https://arxiv.org/abs/2506.07570 |
| SceneWeaver | SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent | NeurIPS 2025 | https://arxiv.org/abs/2509.20414 |
| IL3D | IL3D: A Large-Scale Indoor Layout Dataset for LLM-Driven 3D Scene Generation | arXiv 2025 | https://arxiv.org/abs/2510.12095 |
| SceneSmith | SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes (Pfaff, Cohn, Zakharov, Cory, Tedrake) | ICML 2026 (arXiv 2602.09153) | https://arxiv.org/abs/2602.09153 (html v1 + pdf) |
| SAGE | SAGE: Scalable Agentic 3D Scene Generation for Embodied AI (NVIDIA) | CVPR 2026 | https://arxiv.org/abs/2602.10116 |
| ZoneMaestro | Orchestrating Spatial Semantics via a Zone-Graph Paradigm for Intricate Indoor Scene Generation | arXiv 2026 | https://arxiv.org/abs/2605.02537 |
| GraphValid | Global Graph-Validated Optimization for VLM-based 3D Indoor Scene Generation | arXiv 2026 | https://arxiv.org/abs/2608.03064 |
| ScenePilot | ScenePilot: Grow-and-Repair Policy for Text-Driven 3D Indoor Scene Generation | arXiv 2026 | https://arxiv.org/abs/2608.30307 |
| RoomFormer | Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries | CVPR 2023 | https://arxiv.org/abs/2211.15658 |

Not covered: Ctrl-Room, SceneLCM and Sceniris were downloaded, but I did not read their evaluation sections closely enough to define their metrics. RoomFormer does floorplan reconstruction, not generation. Its metrics (precision/recall/F1 at the room, corner and angle levels after IoU matching; door/window counted correct at L2 < 10 px) are listed only for completeness.

---

## 1. Metric × paper table

Legend: ● = reported as a quantitative metric, H = human study, V = VLM/LLM judge.

### 1a. Distribution / perceptual (compare against a reference dataset; 3D-FRONT-trained models)

| Metric | Papers |
|---|---|
| FID (top-down render) | ATISS, LEGO-Net, LayoutGPT (4 camera views), CommonScenes, EchoScene, DiffuScene, InstructScene, PhyScene, MiDiffusion, FurniScene, DeBaRA, ReSpace, OptiScene |
| KID | LEGO-Net, CommonScenes, EchoScene, DiffuScene, InstructScene, PhyScene, MiDiffusion, FurniScene, DeBaRA, ReSpace |
| FID_CLIP | EchoScene, InstructScene, ReSpace |
| SCA (scene classification accuracy, target 50%) | ATISS, DiffuScene, InstructScene, PhyScene, MiDiffusion ("CA%"), FurniScene, DeBaRA |
| CKL (category KL divergence) | ATISS, LayoutGPT, DiffuScene, PhyScene, MiDiffusion, FurniScene |
| Inception Score | SceneCraft-layout |
| EMD to GT / distance moved | LEGO-Net (rearrangement) |
| Integer-relation regularity | LEGO-Net (new) |

### 1b. Text / prompt adherence (automatic, non-VLM-judge)

| Metric | Papers |
|---|---|
| CLIP score (top-down vs "a top-down view of [scene type]") | Holodeck, GLTS, DirectLayout |
| CLIP score (render vs prompt/caption) | SceneCraft-agent, AnyHome (CLIP/H-14, + CLIP-style-sim), Architect, Scenethesis, HSM (LongCLIP), SceneEval (Long-CLIP-L, as a reference baseline), IL3D (vs reference scene) |
| BLIP(-2) ITM score | Architect, Scenethesis, HSM |
| VQAScore | Architect, Scenethesis, HSM |
| Caption similarity | AnyHome |
| iRecall (instruction triplet recall) | InstructScene |
| Scene-graph constraint accuracy | CommonScenes, EchoScene |
| Object-count precision/recall/accuracy, mIoU vs GT boxes | LayoutGPT (2D), SceneTeller |
| Constraint-passing score (human-written checker functions) | SceneCraft-agent |
| SceneEval CNT / ATR / OOR / OAR | SceneEval, HSM, SceneSmith |
| Prompt Matching Score (PMS) | ReSpace |
| Min/Mean L2 to GT placement, Energy score, Visibility | FirePlace |

### 1c. Physical plausibility / functionality

| Metric | Papers |
|---|---|
| Out-of-bounds (scene-level % of scenes) | LayoutGPT, I-Design, SceneTeller, AnyHome |
| Out-of-bounds (object-level %, area or volume) | PhyScene (R_out), MiDiffusion (OOB%, 0.1 m dilation), DeBaRA (OBA, m²), FlairGPT (area/room area), SceneEval/HSM/SceneSmith (ray-cast 99%), SceneWeaver/SAGE-style (#OB count), DirectLayout, IL3D, ZoneMaestro (volume), ReSpace/ScenePilot (voxel OOB) |
| In-Boundary score (IB) | LayoutVLM, GraphValid, DirectLayout |
| Bounding-box overlap (BBL / PIoU / IoU / OOR overlap rate) | DiffuScene (PIoU), MiDiffusion (IoU%), I-Design (BBL), FlairGPT (OOR), OptiScene (OOR), IL3D (OOR) |
| Mesh collision rate (object-level and/or scene-level) | PhyScene (Col_obj, Col_scene), Scenethesis (Col-O, Col-S), SceneEval/HSM (COL), SAGE (trimesh), DirectLayout, ZoneMaestro (volume) |
| Collision-free score (CF) | LayoutVLM, GraphValid |
| Collided pairs count (#CN) | SceneWeaver |
| Collision with penetration-depth threshold (signed distance, > 1 mm) | SceneSmith (COL, Drake) |
| Voxel-based loss (VBL = OOB + MBL) | ReSpace (new); ScenePilot (PBL, + valid-scene ratio) |
| Walkable / navigable area ratio (largest connected free area / total free area) | PhyScene (R_walkable), Scenethesis (Walk), SceneEval/HSM/SceneSmith (NAV) |
| Object reachability | PhyScene (R_reach), Scenethesis (Reach) |
| Accessibility of functional sides | SceneEval/HSM/SceneSmith (ACC), FlairGPT (C_pathway) |
| Support (ray-cast, VLM-typed) | SceneEval, HSM (SUP) |
| Physics-sim stability (displacement after settling) | Scenethesis (Inst-O/Inst-S), SceneWeaver (Isaac Sim, appendix), SAGE (Isaac Sim, Stab.%), SceneSmith (STB, Drake) |
| Symmetric pairs (Sym), #Obj | DiffuScene (Sym, Obj), MiDiffusion, I-Design (NObj), SceneWeaver, SAGE, SceneSmith, ZoneMaestro (Cnt) |
| Generation success rate | IL3D (GSR), ZoneMaestro (Succ) |
| Downstream embodied-AI task performance (ObjectNav SR/SPL, etc.) | ProcTHOR, Holodeck (zero-shot ObjectNav), SceneSmith (robot policy evaluation), SAGE (policy training) |

### 1d. VLM/LLM-as-judge (V)

| Judge protocol | Papers |
|---|---|
| GPT-4V 0–10, Func./Layout/Scheme/Atmos., 2 corner renders, 3 samples averaged | I-Design |
| GPT-4o 0–100 Positional / Rotational coherency, PSA (physics-weighted) | LayoutVLM, DirectLayout, GraphValid |
| GPT-4 0–10 **Realism** / Functionality / Layout / Completion (top-down render + layout) | SceneWeaver |
| GPT-4.1 **Realism** / Functionality / Layout / Completeness + physics | SAGE |
| GPT-4o pairwise, object diversity / layout coherence / **spatial realism** & complexity / overall | Scenethesis |
| GPT-4o ranking | Architect |
| Gemini 1–4 plausibility score | FirePlace |
| GPT-4o 0–10 Functionality / Furniture Layout / Aesthetics | OptiScene |
| GPT-4o Object Pose / **Physical Reality** / Semantic Consistency / Scene Functionality / Visual Aesthetics | IL3D |
| GPT-4o-mini 0–10 Aesthetics / **Lived-in Realism** / Structure / Geometry / Semantics / Functionality | ZoneMaestro |
| GPT-5.4 layout correctness / semantic plausibility / functional completeness | ScenePilot |
| GPT-4o as component inside rule metrics (object matching, attributes, functional sides, support type) | SceneEval, HSM, SceneSmith |

### 1e. Human studies (H)

| Protocol | Papers |
|---|---|
| 2AFC "more realistic" (+ error frequency) | SceneFormer (AMT, 30 users), ATISS (AMT), Infinigen-Indoors (AMT, + **Layout Realism**), Open-Universe (35 students), I-Design (Bradley–Terry), SceneSmith (Prolific, 205) |
| Pairwise with other criteria | DiffuScene, CommonScenes ("Global correctness and realism"), Holodeck (asset selection / layout coherence / overall), HSM (fidelity / plausibility), FlairGPT, SceneWeaver (preference + diversity), Scenethesis (69 users), ReSpace |
| Likert ratings | Set-the-Scene (**Realism Score**/rank), SceneTeller (**Realism** 1–5), Holodeck (1–5 on 52 scene types), Architect (VQ/SC/LC/OP 1–5), SceneWeaver (Real./Func./Lay./Comp.), FlairGPT, DirectLayout (25 experts), OptiScene (Usability Rate) |
| Ranking (reciprocal rank / Kendall τ vs VLM) | GLTS (MRR, "more realistic"), LayoutVLM, SceneWeaver (Kendall τ human–LLM), Holodeck (MRR ablation), ZoneMaestro |

---

## 2. Per-metric definitions

### 2.1 Distribution metrics (FID / KID / FID_CLIP / SCA / CKL): the "ATISS protocol"

- **ATISS** (https://arxiv.org/abs/2110.03675): "To measure the realism of the generated scenes, we follow prior work [FastSynth] and report the KL divergence between the object category distributions of synthesized and real scenes ... and the classification accuracy of a classifier trained to discriminate real from synthetic scenes. We also report the FID between top-down orthographic projections of synthesized and real scenes ..., which we compute using [clean-fid] on 256² images." Supplementary: they generate as many scenes as the test set has. **SCA** uses "an AlexNet pre-trained on ImageNet, that takes as input a 256² top-down image-based representation of a room and predicts whether this scene is real or synthetic." FID and SCA are repeated 10× and averaged.
- **DiffuScene** (https://arxiv.org/abs/2303.14207): FID, KID×0.001, SCA, CKL×0.01 on 1,000 synthesized scenes. Generated and GT scenes are rendered "into 256×256 semantic maps through top-down orthographic projections, where the texture of each object is uniquely determined by the associate color of its semantic class". SCA is best near 50%. DiffuScene also adds **Obj** (mean object count), **Sym** (mean symmetric pairs per scene) and **PIoU** (pairwise bbox IoU ×0.01).
- **InstructScene** (https://arxiv.org/abs/2402.04717, App. B.3): Blender CYCLES, top-down 256×256, camera distance 1.2, fixed floor textures. FID, **FID_CLIP** (FID on CLIP features, Kynkäänniemi 2022) and KID come from clean-fid. SCA is a fine-tuned ImageNet AlexNet. Results are averaged over 5 seeds.
- **MiDiffusion** (https://arxiv.org/abs/2405.21066): 1,000 test floor plans. Each image covers 6.2 m (bedroom) or 12.2 m (living/dining), with semantic-class coloring. Also reports Obj, OOB% (room boundary dilated 0.1 m) and bbox IoU%.
- **DeBaRA** (https://arxiv.org/abs/2409.18336): FID, KID×1000, SCA on top-down semantic renders, plus **OBA**: "cumulated out of bounds objects area (in m²)".
- **LayoutGPT** (https://arxiv.org/abs/2305.15393): category KL, and FID on renders "from four camera angles for each scene".
- **CommonScenes / EchoScene**: FID/KID (EchoScene also FID_CLIP) on bird's-eye projections "excluding lamps to prevent occlusion and using object colors as semantic indicators". Also reported: scene-graph constraint accuracy, i.e. "the accuracy of a set of relations on a generated layout" (left/right, front/behind, bigger/smaller, ...).
- **LEGO-Net** (https://arxiv.org/abs/2301.09629): FID/KID on top-down renders ("KID is more applicable ... because FID is known to present huge bias when the number of data is low"), distance moved, and EMD to GT. It introduces a **regularity metric**: pick 2–3 random objects and use PSLQ to test whether integer coefficients a_i exist with Σ a_i t_i = 0 and 0 < |a_i| < η. The result is reported as an integer-relation chance.
- **Weaknesses** (as the papers themselves note):
  - These metrics need a reference dataset (3D-FRONT), so they don't apply to open-vocabulary or VLM methods.
  - They depend on the renderer and asset textures.
  - They are uninterpretable. SceneEval: "the higher FID or lower CLIPsim of DiffuScene does not illuminate whether this is due to poor scene layout, unrealistic object appearances, or other visual inconsistencies ... These image-based metrics are sensitive to rendering, making them unreliable when comparing methods that use different asset sources."
  - PhyScene: "these realism metrics do not address the physical plausibility and interactivity of generated scenes."
  - SCA and FID mostly reward matching 3D-FRONT statistics, including its flaws. Per LayoutGPT, 3D-FRONT GT itself has a nonzero OOB rate, and per SceneTeller the GT OOB rate is 15.15%.

### 2.2 Text–image alignment scores

- **CLIP Score (Holodeck)** (https://arxiv.org/abs/2312.09067): "visual coherence between the top-down view of the scene and its corresponding scene type embedded in a prompt template 'a top-down view of [scene type]'". They use "OpenCLIP with ViT-L/14 trained on LAION-2B" and "cosine similarity times 100". iTHOR human-designed scenes serve as the upper reference. GLTS reuses exactly this setup.
- **CLIPScore / BLIPScore / VQAScore (Architect)** (https://arxiv.org/abs/2411.09823): CLIP image–text feature correlation. BLIPScore uses the "image-text-matching head of BLIPv2". VQAScore feeds the render and caption to a VQA model and takes "the probability that the answer to the question 'Does the image show caption' is 'Yes'". Scenethesis and HSM use the same three scores (HSM uses BLIP-2, LongCLIP and VQAScore on top-down renders).
- **AnyHome**: CLIP-sim (CLIP/H-14 on rendered frames vs the input text), CLIP-style-sim, and Caption-sim (caption the frames, then sentence similarity).
- **Weaknesses**: dominated by asset appearance and texture. Insensitive to layout errors such as orientation, collisions and support. Top-down views lose height information. CLIP has a 77-token limit, hence LongCLIP. SceneEval shows a +1.2 CLIPsim gap is "hard to interpret".

### 2.3 Structured prompt-adherence metrics

- **iRecall (InstructScene)**: "the proportion of the required triplets '(subject, relation, object)' occurring in synthesized scenes to all provided in instructions". The relations are rule-based (distance and relative orientation).
- **SceneEval fidelity metrics** (https://arxiv.org/abs/2503.14756; SceneEval-500 with human-annotated scene properties; all VLM calls use GPT-4o-2024-08-06):
  - **Object matching**: each object's front view is rendered, and a VLM matches it to the annotated categories.
  - **CNT**: % of count specifications satisfied (exact or relative quantities).
  - **ATR**: two renders per object (front view, plus a view "with a 170 cm human figure for scale"). The VLM judges whether the annotated attributes hold.
  - **OOR**: 13 relation types (inside, side of, next to, ...). The VLM maps open-vocabulary relations to these types, and geometry (ray casting, point sampling, position analysis) verifies them.
  - **OAR**: 10 object–architecture relation types (against wall, corner, hang ceiling, ...).
  - **Human agreement**: 89.8 / 83.5 / 94.6 / 94.1% (Cohen's κ 0.75 / 0.56 / 0.72 / 0.77) for CNT/ATR/OOR/OAR. A separate 10-person user study gave κ 0.56–0.72.
- **PMS (ReSpace)**: PMS(p, d) = (1/|p|) Σ_{w∈p} 1[w ∈ d]. This is the word recall of the object prompt in the retrieved asset's description.
- **SceneCraft-agent**: CLIP similarity plus a "constraint passing score" from checker functions written by human annotators.
- **Weaknesses**: needs annotated prompts. VLM steps introduce false positives and negatives; SceneSmith notes this explicitly (App. M.2). The relation vocabulary is fixed.

### 2.4 Physical plausibility

- **Out-of-bounds**:
  - Scene-level (LayoutGPT, I-Design, SceneTeller): "the percentage of scenes with furniture exceeding the floor plan boundary". In I-Design, "if any bounding boxes are outside ... that scene is deemed invalid".
  - Object-level: PhyScene R_out is the "rate of objects outside the floor plan".
  - SceneEval OOB: sample surface points and cast rays to the floor; the object is out-of-bounds "if fewer than 99% of the points intersect the floor".
  - Area/volume variants: DeBaRA OBA (m²); FlairGPT OOB = Σ_i A_bound[i] / (w·l); ZoneMaestro OOB volume.
  - Voxel variant (ReSpace): voxelize the boundary mesh and each object mesh, then count object voxels outside the scene boundary.
- **Collision / overlap**:
  - I-Design BBL: "average volume of bounding box intersections".
  - DiffuScene PIoU: pairwise bbox IoU.
  - FlairGPT OOR: overlap area / room area, with door-swing buffers.
  - PhyScene (https://arxiv.org/abs/2404.09465): mesh collisions with CAD models (re-meshed, since 3D-FUTURE meshes are not watertight). **Col_obj** is the % of objects colliding with another object; **Col_scene** is the % of scenes with any collision.
  - SceneEval COL: mesh-based tests between all object pairs, reported as % of objects in collision.
  - SceneWeaver #CN: number of collided object pairs.
  - LayoutVLM CF: Collision-Free score, reported 0–100.
  - **SceneSmith COL**: Drake signed-distance queries on collision geometry; the "fraction of objects in collision with penetration depth exceeding 1mm". Baseline scenes get V-HACD hulls (128 max) first.
  - **ReSpace VBL** (https://arxiv.org/abs/2506.02459): VBL = OOB + MBL, where MBL is mesh-overlap voxels between objects. ReSpace argues bbox metrics "cannot accurately evaluate realistic object" interactions such as a chair tucked under a table.
- **Weaknesses**:
  - Bbox-based collision penalizes legitimate interpenetration of boxes (chair under table) and misses mesh penetrations.
  - Binary scene-level rates saturate.
  - A zero-collision score can be gamed by sparse scenes or shrunken objects. ReSpace notes reward hacking: "shrinking object sizes to trivially lower VBL".
  - SceneEval: an empty room can score perfectly on plausibility, hence its OOB check and the fidelity metrics.

### 2.5 Navigability, reachability, accessibility

- **PhyScene R_walkable**: "average ratio of the largest connected walkable area over all walkable areas in the room". The 2D room mask is computed considering agent size.
- **PhyScene R_reach**: "average reachable rate of objects in the scene starting from a random starting point on the floor plan".
- **SceneEval NAV**: "ratio of the largest connected free space to the total free space", computed via 2D occupancy projection and connected components. Used unchanged by HSM and SceneSmith.
- **SceneEval ACC**: a VLM identifies each object's functional sides; 2D occupancy analysis then gives "the ratio of unoccupied pixels to the total functional area, taking the best score if multiple sides exist".
- **FlairGPT C_pathway**: a pathway clearance cost based on distance from pathway points to the nearest object boundary.
- **Weaknesses**: 2D projections ignore overhangs (tables you can walk partly under). Results are sensitive to agent radius and resolution. Denser scenes (SceneSmith, with 71 objects per room) score lower on NAV and ACC. SceneSmith explicitly reports "lower ACC and NAV scores" as a trade-off of density.

### 2.6 Support and stability (physics-simulator metrics, new in 2024–2026)

- **SceneEval SUP**: a VLM classifies each object's support type (ground / object / wall / ceiling), then ray casting checks the support. Reported as % correctly supported.
- **Scenethesis** (https://arxiv.org/abs/2505.02836): **Inst-O** (object instability rate) and **Inst-S** (scene instability rate). "Instability follows Atlas3D ..., measured by tracking transformations after physics-based simulation". The thresholds are not stated in the main text. Collisions Col-O/Col-S use "mesh-mesh intersections".
- **SceneWeaver** (App. B.6, Isaac Sim): "percentage of objects moving >0.1m or >0.01m in 3s" and "average displacement".
- **SAGE** (https://arxiv.org/abs/2602.10116): Stab.% is the ratio of stable objects in Isaac Sim. An object is "unstable if its relative translation exceeds 0.2 meter or its rotation exceeds 8 degrees after 120 simulation steps". Coll.% is the mesh collision ratio via trimesh.
- **SceneSmith STB**: "Runs a 5-second physics simulation with gravity (1ms time step). Objects are marked stable if displacement < 1cm and rotation < 0.1rad after settling. Wall and ceiling objects are welded". Support types come from SceneEval's VLM. Baselines get V-HACD collision geometry and mass/inertia "assuming a uniform density of 1000 kg/m³". Carpets are excluded (following SceneWeaver).
- **Weaknesses**: thresholds differ by 20× across papers (1 cm vs 0.2 m), so numbers are not comparable. Results depend on collision proxies (convex decomposition), assumed densities and friction, and on how baselines are converted. Stability can be enforced by the generator itself (SceneSmith settles scenes in Drake before evaluation).

### 2.7 VLM/LLM-as-judge (spread from 2024 onward)

- **I-Design** (https://arxiv.org/abs/2404.02838): GPT-4V gives integer 0–10 grades on Func., Layout, Scheme and Atmos. Input is "Two renderings from different viewpoints ... horizontally concatenated", plus the user input. The judge runs 3 times (API n=3) and reports mean ± std. The appendix system prompt also lists "Realism and 3D Geometric Consistency" as an aspect, but the table reports only the four categories above. A chain-of-thought comment precedes each grade.
- **LayoutVLM** (https://arxiv.org/abs/2412.02193): GPT-4o scores **Positional coherency** and **Rotational coherency** (0–100) from top-down and side-view renders plus the instruction. "PSA is calculated simply the GPT-4o rating weighted by physical plausibility", and set to 0 if assets cannot be placed. The rubric uses bands such as "100-81: Excellently Positioned". Validation: 5 graduate students ranked the methods, and Kendall's τ against GPT-4o was reported.
- **SceneWeaver** (https://arxiv.org/abs/2509.20414): "visual realism (Real.), functionality (Func.), layout correctness (Lay.), and scene completeness (Comp.) ... we use GPT-4 to assess these metrics, providing it with top-down renderings ... and the user query". The verifier prompt (Table A2) is 0–10 per criterion. Kendall's τ is reported between users and LLM.
- **Scenethesis**: pairwise GPT-4o comparison of image pairs (2 views per scene, 22 scenes → 44 pairs). Output: "{} is better than {} in the criteria {}". It is told to ignore lighting and decoration.
- **FirePlace**: Gemini plausibility score from 1 (poor) to 4 (great) with a rubric, validated against a user study.
- **OptiScene**: GPT-4o 0–10 on Functionality, Furniture Layout and Aesthetics. Also **Usability Rate**: of 50 scenes × 5 annotators, a scene is usable if ≥3 judge it physically and semantically valid.
- **IL3D**: GPT ratings on Object Pose, **Physical Reality**, Semantic Consistency, Scene Functionality and Visual Aesthetics.
- **ZoneMaestro** (https://arxiv.org/abs/2605.02537): GPT-4o-mini integer 0–10 on six dimensions (the main text says 1–10). The judge is told to "IGNORE Rendering Quality" and asset texture.
- **ScenePilot** (https://arxiv.org/abs/2608.30307): GPT-5.4 scores LC / SPA / FC from top and diagonal views. Mirrored by a human study with the same criteria.
- **Weaknesses**:
  - Judge drift across model versions (GPT-4V → 4o → 4.1 → 5.x), so scores are not comparable across papers.
  - Sensitivity to the render viewpoint (top-down hides vertical errors) and to texture and lighting.
  - Scale compression: scores cluster around 6–9.
  - Self-preference when the same model family generates and judges. SceneWeaver and SceneSmith use GPT critics in the generation loop and GPT-family judges for evaluation.
  - Prompts are rarely shared across papers.
  - SceneSmith shows that swapping the baselines' LLM to GPT-5.2 changes results inconsistently.

### 2.8 Human studies

- **ATISS**: paired AMT study; users see rotating 3D scenes. Questions cover realism ("more realistic") and error frequency (e.g. interpenetrating furniture). Binomial 99% CIs are reported.
- **Infinigen Indoors** adds **Layout Realism**: "asked the subjects to focus only on the arrangement of the furniture and ignore the style of individual objects".
- **SceneSmith**: Prolific, 205 participants, 3,051 responses, pairwise in an interactive 3D viewer, with Wilson CIs, Benjamini–Hochberg correction and Cohen's h.
- **Weaknesses**: small N, mostly students, and every paper uses its own protocol. "Realism" judgments conflate rendering, asset and lighting quality with layout. Infinigen Indoors says so explicitly: "'realism' may be influenced by asset and lighting quality".

### 2.9 Downstream / task-based

- **ProcTHOR**: models are trained on ProcTHOR-10K and evaluated on ObjectNav, Rearrangement and ArmPointNav (success rate, SPL), including zero-shot.
- **Holodeck**: zero-shot ObjectNav on NoveltyTHOR.
- **SceneSmith**: robot policy evaluation on 100 scenes × 4 pick-and-place tasks. The evaluator agent agreed with humans on 299/300 judgments.
- **SAGE**: policies trained purely on generated data show scaling trends.
- **Infinigen Indoors**: synthetic training data for occlusion-boundary detection and shadow removal.

---

## 3. "Realism" — which indoor-scene papers report a metric explicitly called realism, and how it is computed

**VLM/LLM-judged realism**

1. **SceneWeaver** (NeurIPS 2025): **"Real." (visual realism)**, scored 0–10 by GPT-4 from a top-down render, the layout and the user query. Rubric: "Realism: How realistic the room appears. Ignore texture, lighting, and doors. Good (8-10): The layout (position, rotation, and size) is believable, and common daily objects make the room feel lived-in ... Bad (0-3): Unusual objects or strange placements ... If object types or combinations defy real-world logic (e.g., bathtubs in bedrooms), score should be below 5." It sits next to Func., Lay. and Comp. A human study with 20 participants used the same four criteria, and Kendall's τ checked human–LLM alignment. **This is the most likely candidate for "a paper with a realism metric".**
2. **SAGE** (CVPR 2026): **"Real."** with Func./Lay./Comp. from **GPT-4.1**, using the SceneWeaver-style visual-metric set, alongside Coll.% (trimesh) and Stab.% (Isaac Sim, 0.2 m / 8° after 120 steps).
3. **Scenethesis** (ICLR 2026): a "Layout Realism" category containing **"Spatial Realism"**, defined as the "presence of diverse spatial relations (e.g., on top of, inside, under)". It is judged pairwise by GPT-4o ("Spatial realism & complexity") and by 69 human users; the result is a preference % for Scenethesis vs each baseline.
4. **ZoneMaestro** (arXiv 2026): **"Realism (Real) for lived-in plausibility"**, scored by GPT-4o-mini 0–10: "Does the scene look like a real, inhabited space with natural 'clutter' and organic variation? Or ... a sterile, artificial AI-generated showroom with rigid, grid-like alignment?"
5. **IL3D** (arXiv 2025): **"Physical Reality (PR)"**, a GPT rating that "checks the physical authenticity of the scene". The name is realism-adjacent, not literally "realism".
6. **I-Design** (ECCV-W 2024): the GPT-4V system prompt includes "Realism and 3D Geometric Consistency", but the reported table shows only Func./Layout/Scheme/Atmos. The human study asked users to "vote on the most realistic room", and the results were converted to Bradley–Terry scores.

**Human-judged realism**

7. **SceneSmith** (ICML 2026): **Realism win rate**, a pairwise forced choice: "Which scene looks more realistic?" with the hint "Consider: realism, object placement, layout coherence". 205 Prolific participants, 15 comparisons each, an interactive 3D viewer, blinded and randomized. Reported as win rate with 95% Wilson CIs (92.2% average vs baselines). No formal definition beyond the tooltip.
8. **ATISS** (NeurIPS 2021): AMT A/B test reporting "More Realistic" fraction with 99% binomial CI, plus "Mean Error Frequency".
9. **Infinigen Indoors** (CVPR 2024): AMT "Realism" and **"Layout Realism"** (arrangement only, ignore object style), following ATISS. Compared against ProcTHOR, ATISS, SceneFormer and FastSynth.
10. **SceneFormer** (3DV 2021): AMT, 30 users, 50 pairs each; "The user must select the more realistic scene", with 5 vigilance pairs.
11. **Open-Universe** (2024): 2AFC with 35 students; "choose which scene they thought was a more realistic instance of that type of room" (79% vs ATISS, 81% vs DiffuScene).
12. **SceneTeller** (ECCV 2024): 30 participants rate **"Realism"** 1–5, along with Text Alignment, Geometric Fidelity and Compositional Plausibility.
13. **Set-the-Scene** (ICCV 2023): **"Realism Rank"** and **"Realism Score"** (1–5) from a user study on NeRF scenes.
14. **CommonScenes** (NeurIPS 2023): user-study criterion "Global correctness and realism (Does the arrangement of objects look correct?)".
15. **GLTS** (CVPR 2025): 15 annotators rank scenes on "Which scene is more realistic and makes common sense, jointly considering the position and orientation?", reported as reciprocal rank.
16. **Holodeck** (CVPR 2024): "Layout Coherence: which arrangement of 3D assets adheres better to realism and common sense". The metric is not named "realism".

**Note on terminology.** ATISS, DiffuScene, DeBaRA and PhyScene all call FID/KID/SCA/CKL "realism" metrics ("To measure the realism of the generated scenes ... KL ... classification accuracy ... FID"). If the user remembers a "realism" column next to FID, that is this distributional usage.

---

## 4. Trends

1. **2019–2023 (3D-FRONT era)**: the ATISS protocol dominates (FID/KID on 256² top-down semantic renders, SCA with AlexNet, CKL), plus AMT 2AFC realism studies. LayoutGPT adds a scene-level OOB rate. DiffuScene adds PIoU, Sym and #Obj. CommonScenes and EchoScene add FID_CLIP and scene-graph constraint accuracy. InstructScene adds iRecall.
2. **2024 (physics and embodiment)**:
   - PhyScene adds the standard physical suite: Col_obj, Col_scene (mesh), R_out, R_reach, R_walkable.
   - Open-vocabulary LLM methods (Holodeck, I-Design, AnyHome, Architect) cannot use FID. They move to CLIP score, BLIP/VQAScore, and VLM judges (I-Design's GPT-4V 0–10 protocol).
3. **2025 (benchmarks and VLM judges)**:
   - LayoutVLM's CF/IB/Pos./Rot./PSA (GPT-4o 0–100) is reused by DirectLayout and GraphValid.
   - SceneEval (WACV 2026) offers interpretable, annotation-grounded fidelity metrics (CNT/ATR/OOR/OAR) and plausibility metrics (COL/SUP/NAV/ACC/OOB), and explicitly argues against FID, KID, SCA and CLIPsim. HSM and SceneSmith adopt it.
   - ReSpace introduces voxel-based VBL; ScenePilot adopts it.
4. **2025–2026 (agentic, simulation-ready)**:
   - **Physics-simulator stability** is new and spreading: Scenethesis (Atlas3D-style), SceneWeaver (Isaac Sim), SAGE (Isaac Sim), SceneSmith (Drake STB).
   - Collision is measured with **penetration depth** (SceneSmith, 1 mm) rather than bbox overlap.
   - Object density (#Obj) is now a headline metric.
   - **VLM-as-judge "Realism / Functionality / Layout / Completeness"** (SceneWeaver → SAGE) became a de facto agentic-scene suite.
   - Large pairwise **human realism/faithfulness win-rate** studies with proper statistics (SceneSmith).
   - **Downstream robot-policy** evaluation (SceneSmith, SAGE).
   - Judge models drift (GPT-4V → 4o → 4.1 → 5.2 → 5.4), so absolute scores are incomparable across papers.
5. **Standard now**:
   - (a) OOB rate.
   - (b) Collision rate (object-level, mesh).
   - (c) Navigability (largest connected free area ratio).
   - (d) #Obj.
   - (e) A VLM-judge score (realism/functionality/layout).
   - (f) A pairwise human study.
   - FID/KID/SCA/CKL remain only for closed-vocabulary 3D-FRONT methods.
   - SceneEval-100/500 is becoming the shared prompt set for text-conditioned methods.

---

## 5. What SceneSmith uses (summary)

Prompts: 210 total, of which 179 are room-level (SceneEval-100 rooms, Type Diversity, Object Density, Themed) and 31 house-level.

- **Human study**: pairwise realism (forced choice) and prompt faithfulness (A/B/Equal, ties excluded). Reported as win rates with Wilson CIs, BH correction and Cohen's h. 205 Prolific participants.
- **SceneEval metrics**: CNT, ATR, OOR, OAR, ACC, NAV, OOB.
- **#Obj**: objects per room (71.1 for SceneSmith vs 11–23 for baselines).
- **Physics metrics (new)**:
  - **COL**: Drake signed distance, penetration > 1 mm.
  - **STB**: 5 s at a 1 ms step; stable if displacement < 1 cm and rotation < 0.1 rad.
  - Baselines are converted with V-HACD (128 hulls) at 1000 kg/m³; carpets excluded.
- **Downstream**: robot policy evaluation with an evaluator agent (99.7% agreement with humans).
- **Not used**: FID, KID, SCA or CLIP score.
