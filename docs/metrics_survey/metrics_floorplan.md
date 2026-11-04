# Evaluation metrics in floor plan / residential layout generation (2019–2026)

Compiled 2026-10-21. Every definition below was read from the paper's full text (arXiv PDF, CVF open access, ACL Anthology or author-hosted PDF, extracted with `pdftotext`). Where I could not get the full text, the entry is marked **UNVERIFIED**.

## 0. Papers covered (with the source I read)

| Short name | Title / venue | Source read |
|---|---|---|
| RPLAN | Wu et al., *Data-driven Interior Plan Generation for Residential Buildings*, TOG / SIGGRAPH Asia 2019 | https://wutomwu.github.io/publications/2019-DeepLayout/paper.pdf (project: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/) |
| House-GAN | Nauata et al., ECCV 2020 | https://arxiv.org/abs/2003.06988 |
| Graph2Plan | Hu et al., TOG / SIGGRAPH 2020 | https://arxiv.org/abs/2004.13204 |
| House-GAN++ | Nauata et al., CVPR 2021 | https://arxiv.org/abs/2103.02574 |
| iPLAN | He, Huang, Wang, CVPR 2022 | https://openaccess.thecvf.com/content/CVPR2022/papers/He_iPLAN_Interactive_and_Procedural_Layout_Planning_CVPR_2022_paper.pdf |
| WallPlan | Sun et al., TOG / SIGGRAPH 2022 | https://faculty.hfut.edu.cn/zhenglp/zh_CN/article/154624/content/1214.htm (PDF linked there; also https://wutomwu.github.io/publications/2022-WallPlan/paper.pdf) |
| Panoptic-Refine | Liu et al., *End-to-end Graph-constrained Vectorized Floorplan Generation with Panoptic Refinement*, ECCV 2022 | https://arxiv.org/abs/2207.13268 |
| FloorplanGAN | Luo & Huang, *FloorplanGAN: Vector residential floorplan adversarial generation*, Automation in Construction 142 (2022) | **UNVERIFIED** – paywalled, full text not obtained |
| HouseDiffusion | Shabani, Hosseini, Furukawa, CVPR 2023 | https://arxiv.org/abs/2211.13287 |
| GTGAN | Tang et al., *Graph Transformer GANs for Graph-Constrained House Generation*, CVPR 2023 | https://arxiv.org/abs/2303.08225 |
| Tell2Design (T2D) | Leng et al., ACL 2023 | https://arxiv.org/abs/2311.15941 |
| BubbleFormer | Sun et al., CGF / Pacific Graphics 2023 | https://wutomwu.github.io/publications/2023-BubbleFormer/paper.pdf |
| MSD-HD report | Kuhn, *Adapting HouseDiffusion for conditional floor plan generation on MSD* (MSD challenge report, 2023) | https://arxiv.org/abs/2312.03938 |
| MaskPLAN | Zhang, Savov, Dillenburger, CVPR 2024 | https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_MaskPLAN_Masked_Generative_Layout_Planning_from_Partial_Input_CVPR_2024_paper.html |
| DStruct2Design | Luo et al., arXiv 2024 | https://arxiv.org/abs/2407.15723 |
| MSD | van Engelenburg et al., ECCV 2024 | https://arxiv.org/abs/2407.10121 |
| ChatHouseDiffusion | Qin et al., arXiv 2024 | https://arxiv.org/abs/2410.11908 |
| HouseLLM → HouseTune | Zong et al., arXiv 2411.12279 (v1 titled *HouseLLM: LLM-Assisted Two-Phase Text-to-Floorplan Generation*, renamed *HouseTune* in v4) | https://arxiv.org/abs/2411.12279 |
| GSDiff | Hu, Wu et al., AAAI 2025 | https://arxiv.org/abs/2408.16258 |
| FP-LLaMa | Yin et al., *FloorPlan-LLaMa*, ACL 2025 | https://aclanthology.org/2025.acl-long.331 |
| DiffPlanner | Wang & Pajarola, TVCG 2025 | https://arxiv.org/abs/2508.13738 |
| FloorplanSBS | Wu et al., ACM MM 2025 | https://wutomwu.github.io/publications/2025-FloorplanSBS/paper.pdf |
| GreenPlanner | Zeng et al., arXiv 2512.00406 (Dec 2025) | https://arxiv.org/abs/2512.00406 |
| CE2EPlan | Wang & Pajarola, *Directly from Alpha to Omega*, TVCG 2026 | https://arxiv.org/abs/2602.20377 |
| BCA/DS | Stoppani et al., *Boundary-Constrained Diffusion Models for Floorplan Generation: Balancing Realism and Diversity*, ESANN 2026 | https://arxiv.org/abs/2602.01949 |
| FloorplanGSO | Wu et al., CGF / Eurographics 2026 | https://wutomwu.github.io/publications/2026-FloorplanGSO/paper.pdf |
| HouseMind | Qin, Weber, Lu, CVPR 2026 | https://arxiv.org/abs/2603.11640 |
| FMLM | Shiohara & Yamasaki, *Unified Vector Floorplan Generation via Markup Representation*, CVPR 2026 | https://arxiv.org/abs/2604.04859 |
| Floor-plan RLVR | Lara et al., *Generative Floor Plan Design with LLMs via RLVR*, Findings of ACL 2026 | https://arxiv.org/abs/2605.14117 |
| GRE-Diff | Wang et al., *GRE-Diff: Gaussian Room Embeddings for Structured Layout Diffusion*, arXiv 2607.08086 (2026; you listed it as ECCV 2026, but arXiv says only "conference", in LNCS format) | https://arxiv.org/abs/2607.08086 |

Also checked: DPLAN (arXiv 2606.21159) is a graph-theoretic construction method with no learned-generation metrics, so it is not tabulated.

---

## 1. Metric × paper table

Legend: ● = reported as a headline metric; ○ = reported in ablation, appendix or secondary analysis. "Realism (US)" = user-study realism score.

| Metric (category) | RPLAN 19 | HG 20 | G2P 20 | HG++ 21 | iPLAN 22 | WallPlan 22 | Pan-Ref 22 | HD 23 | GTGAN 23 | T2D 23 | BubbleF 23 | MaskPLAN 24 | DS2D 24 | MSD 24 | ChatHD 24 | HouseTune 24/25 | GSDiff 25 | FP-LLaMa 25 | DiffPlanner 25 | SBS 25 | GreenPl 25 | CE2E 26 | BCA/DS 26 | GSO 26 | HouseMind 26 | FMLM 26 | RLVR 26 | GRE-Diff 26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Realism (pairwise user-study score, −1…+1)** (user study) | | ● | | ● | | | ● (rank) | ● | ● | | | | | | | ● | | | | | | | | | | | ● | |
| Plausibility / preference forced-choice (vote %) (user study) | ● | | ● | | | ● | | | | | | | | | | | | | ● | | | ● | | | | ● (win rate) | | ● ("Realism") |
| Human Likert rating (alignment / rationality) (user study) | | | | | | | | | | ● | | | | | | | | ● | | | | | | | | | | |
| FID on rendered plans ("Diversity" in HG family) (realism/distribution) | | ● | | ● | ● (FID_img) | ● | ● | ● | ● | | ● | ● | | | | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| KID (distribution) | | | | | | | | | | | | | | | | | ● | | | ● | | | | ● | | | | ● |
| FID on attribute vectors (FID_area, FID_type) (distribution) | | | | | ● | | | | | | | | | | | | | | | | | | | | | | | |
| LPIPS COV / MMD (diversity/coverage) | | | | | | | | | | | | | | | | | | | | ● (MMD) | | | | | | | | ● |
| Diversity Score (trace of feature covariance, per condition) (diversity) | | | | | | | | | | | | | | | | | | | | | | | ● | | | | | |
| Room-count ratio / room-type statistics vs dataset (distribution) | | | | | | | | | | | | | | | | | ● | | | | | | | ○ (histograms) | | | | |
| **Compatibility = GED(input graph, graph of output)** (condition fidelity) | | ● | | ● | | | ● | ● | ● | | | | ● | | | ● | ● | | | | | | ● (GC) | | | ● | ● | |
| GED vs GT graph (not input graph) (fidelity, GT-based) | | | | | | | | | | | | | | | | | | ○ (as motivation) | | | | ● | | | ● | | | |
| Edge-retention compatibility (fraction of target edges kept) (condition fidelity) | | | | | | | | | | | | | | ● | | | | | | | | | | | | | | |
| Node F1 / Edge overlap vs GT graph (fidelity) | | | | | | | | | | | | | | | | | | | | | | | | | ● | | | |
| Micro / Macro IoU vs GT (fidelity, GT-based) | | | | | | | | | | ● | | | | | ● | ● | | | | | ● | | | | ● | | | |
| MIoU over classes vs GT (MSD) (fidelity, GT-based) | | | | | | | | | | | | | | ● | | | | | | | | | | | | | | |
| Box IoU vs GT boxes (fidelity, GT-based) | | | ● | | | | | | | | | | | | | | | | | | | | | | | | | |
| Boundary IoU (interior of boundary vs union of rooms) (constraint satisfaction, GT-free) | | | | | | | | | | | | | | | | | | | | | | | | | | ● | | |
| Boundary Constraint BC / Boundary Compatibility (constraint satisfaction) | | | | | | | | | | | | | | | | | | | | | | | ● | | | | | ● |
| Room Constraint RC (exact type-count match rate), F1_BC-RC (constraint satisfaction) | | | | | | | | | | | | | | | | | | | | | | | | | | | | ● |
| MSE on type-count / adjacency / size vectors (fidelity, GT-based) | | | | | | | | | | | | ● | | | | | | | | | | | | | | | | |
| Self-consistency (stated vs polygon area, room count, duplicate IDs, width/height) (LLM-specific) | | | | | | | | | | | | | ● | | | | | | | | | | | | | | | |
| Prompt-consistency (num rooms, total area, room area, ID/type P/R, ID-type match) (LLM / condition fidelity) | | | | | | | | | | | | | ● | | | | | | | | | | | | | | ● (Room Area MAPE, Room ID acc.) | |
| Overlap (boolean any-overlap) / % Overlap area / overlap-free rate (geometric validity) | | | | | | | | | | | | | ● | | | | | | | | | | | | | | ● | ● |
| Reachability (door graph connected) (functional validity) | | | | | | | | | | | | | | | | | | | | | | | | | | | | ● |
| Functionality (4 basic room types present) / Connectivity (doors per room) (functional validity) | | | | | | | | | | | | | | | | | | | | | | | | ● | | | | |
| Fake-edge ratio, alignment error (geometric validity, ablation) | | | | | | | | | | | | | | | | | ○ | | | | | | | | | | | |
| Functional Rationality (fire distance, area, connectivity) + EUI energy (learned surrogate) (domain validity) | | | | | | | | | | | | | | | | | | | | | ● | | | | | | | |
| ARS – learned architect-preference score (FloorPlan-MPS) (learned judge) | | | | | | | | | | | | | | | | | | ● | | | | | | | | | | |
| SSIM / PSNR vs GT image (pixel fidelity, GT-based) | | | | | | | | | | | | | | | | | | ● | | | | | | | ● | | | |
| LLM parsing accuracy (type / size / location extracted from text) (LLM-specific) | | | | | | | | | | | | | | | ● | | | | | | | | | | | | | |
| Success rate, room match rate, location/adjacency/relation accuracy (understanding tasks) (LLM-specific) | | | | | | | | | | | | | | | | | | | | | | | | | ● | | | |
| Editing metrics (ΔIoU, ΔMSE, Tiny-ROE) | | | | | | | | | | | | | | | | | | | | | | | | | ● | | | ● |
| LPIPS "novelty" to nearest training sample | | | | | | | | | | | | | | | | | | | | | | | | | | | | ○ |

**De-facto standard set.** The standard set is the **House-GAN triple: Realism (user study) / Diversity (FID) / Compatibility (GED to the input bubble diagram)**. It is reported per room-count group (5/6/7/8 rooms on RPLAN, with the held-out-group k-fold protocol) and was introduced by House-GAN (ECCV 2020). It was carried forward by House-GAN++, Panoptic-Refine, HouseDiffusion, GTGAN, HouseTune and RLVR (2026). Most later papers copy the HouseDiffusion Table 1 numbers verbatim instead of re-running the baselines. For boundary-conditioned work (RPLAN, Graph2Plan, WallPlan, iPLAN, DiffPlanner, GSDiff and others), the de-facto set is **FID (+KID) plus a forced-choice plausibility study against GT**. For text-conditioned work (Tell2Design lineage), it is **Micro/Macro IoU vs GT**.

---

## 2. Per-metric definitions

### 2.1 Realism: pairwise user-study score (House-GAN family). Category: user study
- **House-GAN (ECCV 2020)** ([arXiv 2003.06988](https://arxiv.org/abs/2003.06988)): "The realism is measured by an average user rating. We present a generated house layout against a ground-truth or another method. A subject puts one of the four ratings: better (+1), worse (−1), equally-good (+1), or equally-bad (−1)." Participants were 12 graduate students and 10 professional architects, and each subject compared 75 pairs sampled from 5 methods + GT. X = 10 samples per input were used for realism and diversity. The paper reports a single "All groups" realism number per method plus pairwise difference matrices (max difference 2.0). (The "equally-good (+1)" scoring is as printed.)
- **House-GAN++ (CVPR 2021)** ([arXiv 2103.02574](https://arxiv.org/abs/2103.02574)): changes the scale to win/tie/loss. "A subject is presented a pair of layouts … and asked which one is more realistic or a tie. A method scores (+1) for a win, (−1) for a loss, and (0) for a tie. For each pair of methods, we generate 100 pairs to be presented to 10 amateurs and 10 professional architects, where the average score becomes the realism metric." Layouts are shown both as raw masks and as Floor-SP-vectorized plans. The reported number is the average of the pairwise scores over all other methods.
- **HouseDiffusion (CVPR 2023)** ([arXiv 2211.13287](https://arxiv.org/abs/2211.13287)): "present two samples (from different systems) to a participant, and ask to choose 'A is better', 'B is better', or 'both are equal'. A method 'A' earns +1, −1, or 0." Each pair of systems is evaluated 150 times by 10 participants (15 each). **Important change:** HouseDiffusion reports realism **only against GT** ("comparisons between weaker baselines do not provide useful information"). Its −0.19 is read as "as realistic as GT 81% of the time". Realism is skipped for the Non-Manhattan RPLAN benchmark. The table mixes House-GAN++'s "vs-all" average with its own "vs-GT" score, so the column is not apples-to-apples.
- **GTGAN (CVPR 2023)** ([arXiv 2303.08225](https://arxiv.org/abs/2303.08225)): copies the House-GAN protocol: average rating from 12 PhD students + 10 architects over 75 comparisons, ratings better(+1)/equally-good(+1)/worse(−1)/equally-bad(−1).
- **Panoptic-Refine (ECCV 2022)** ([arXiv 2207.13268](https://arxiv.org/abs/2207.13268)): realism is a **ranking** variant. 10 respondents with architecture expertise + 7 amateurs rank 3 methods' outputs for each of 100 bubble diagrams "based on visual quality and functionality". The score is +1 for rank 1, 0 for rank 2 and −1 for rank 3.
- **HouseTune / HouseLLM (arXiv 2024–25)** ([arXiv 2411.12279](https://arxiv.org/abs/2411.12279)): "Realism is an estimate based on a user survey … we follow the same procedure as House-GAN++." The paper reports −0.03 and says "65% of participants perceive our generated layouts as comparable to the ground truth". Participant counts are in the appendix and were not stated in the main text.
- **Floor-plan RLVR (Findings ACL 2026)** ([arXiv 2605.14117](https://arxiv.org/abs/2605.14117)): "each person reviews 10 randomized pairs of layouts (one ground truth, one generated) and selects the more realistic layout, or indicates that both are equally realistic. … +1 if the volunteer selects the generated floor plan, −1 if they select the ground-truth floor plan, and 0 if they judge them equally realistic. We report the mean." There were 40 volunteers and the result is 0.028 (shown as 0.03). The instructions given to volunteers are printed in App. A.6. The score is **GT-anchored**, as in HouseDiffusion.
- **Weaknesses:** the protocols are not comparable across papers (4-level vs 3-level scale, vs-all vs vs-GT, ranking). Panels are small (10–40 people, often students), and inter-rater agreement is never reported. A value near 0 against GT on RPLAN is easy to reach because the GT itself is noisy. Baseline realism numbers are copied rather than re-measured, and the score can't be reproduced or computed automatically.

### 2.2 Forced-choice plausibility / preference vs GT or a competitor. Category: user study
- **RPLAN (2019)**: "Given a pair of floor plans that share the same boundary … each participant should choose the floor plan that they think is more plausible." Each study has 30 forced-choice tasks with a vigilance test (a jumbled random plan) in 1 of every 15 tasks, and 81–99 participants per study (general users vs designers). The paper reports N_avg (number of own results preferred out of 28), where about 14 means parity.
- **Graph2Plan (2020)**: 20 GT plans; 30 CS graduate students; 600 answers; options GT / ours / "Cannot tell" (304/218/78), with filter tasks.
- **WallPlan (2022)**: "evaluate the generation quality and perceived realism". There are 5 groups × 20 pairs with "more plausible" or "Not sure" options, plus a vigilance test; 34 participants were recruited and 28 were valid.
- **DiffPlanner (TVCG 2025)** / **CE2EPlan (TVCG 2026)**: 3 professional designers, 50 GT-vs-ours pairs per group, "GT better / ours better / equally good". DiffPlanner also reports **Fleiss' κ (0.2 floor plans, 0.1 bubble diagrams = slight agreement)**. This is the only paper found that reports inter-rater agreement.
- **FMLM (CVPR 2026)**: 20 AMT users "select the most functional and natural floorplan" among Ours / HouseGAN++ / HouseDiffusion for 100 sets, reported as a win rate (51/24/32%).
- **GRE-Diff (2026)**: section titled **"Realism"**. 58 CS graduate students gave 1,740 responses, choosing their preferred design between GT/Ours/Cannot tell and Graph2Plan/Ours/Cannot tell. The paper reports "preferred our layouts over GT in 60.69% of cases".
- **Weaknesses:** same as above. Note that "preferred over GT >50%" usually reflects GT noise or a style preference rather than super-human quality.

### 2.3 Likert human ratings. Category: user study
- **Tell2Design**: 5 volunteers with NLP backgrounds; 100 instructions; ratings 1–5 on four partial-alignment criteria (room type, location, size, relationships) plus a binary "meets all specifications", for both GT and T2D outputs.
- **FP-LLaMa (ACL 2025)**: 8 architecture PhD students + 5 architects rate "architectural layout rationality" of 150 image sets on a 1–10 scale, supervised by 2 experts. Variances are reported (Table 6).

### 2.4 FID on rendered floor plans. Category: realism/distribution (the House-GAN lineage calls it "Diversity")
- **House-GAN**: "The diversity is measured by the FID score with the rasterized layout images. We rasterize a layout by 1) Setting the background to white; 2) Sorting the rooms in the decreasing order of the areas; and 3) Painting each room with a color based on its room type." The data is at 256×256. Each group uses 5,000 bubble diagrams × 10 samples.
- **House-GAN++ / HouseDiffusion**: FID over 1,000 samples per group, 5 rounds, mean ± std. Copied numbers are used for the baselines.
- **WallPlan / BubbleFormer / DiffPlanner**: FID on 9,000 (WallPlan, BubbleFormer) or 12K (DiffPlanner) generated plans vs GT. The authors note that "the FID score is highly correlated to the example number". Values like 0.46–2.7 are not comparable with the House-GAN-style 5–60.
- **CE2EPlan**: renders vector plans "into 512 × 512 raster images with the same colormap for each competing method", 12K test examples.
- **GRE-Diff**: "Each floorplan is rendered as a 256 × 256 RGB image with a fixed color configuration and 4-pixel wall width". Implemented with torch-fidelity, 512 test layouts, 5 runs.
- **HouseMind**: 256×256 colour-mapped layouts with walls appended.
- **FMLM**: FID on all tasks. For the GSDiff comparison it removes doors from both sides because GSDiff cannot generate them, which shows how much the rendering choice matters.
- **RLVR**: "We compute FID using a custom visualization pipeline that mimics the HouseDiffusion visualizer on our data."
- **iPLAN**: FID_img on rendered images. **MaskPLAN**: fid_img on "standardized" renders.
- **GreenPlanner** and **HouseMind** call FID "visual realism" / "distributional realism"; GSDiff calls it a measure of "quality and diversity".
- **Weaknesses:** (i) Inception features were trained on natural images, but these are flat-colour renders with a few hundred distinct pixel patterns, so the score is dominated by palette, wall thickness, door glyphs and resolution. (ii) There is no shared renderer or resolution: 256 vs 512, with or without walls and doors, and per-paper colour maps, so cross-paper numbers are meaningless. (iii) It needs large sample counts, and the reported absolute values span 0.46 to 180 depending on N. (iv) It is called "Diversity" in House-GAN-lineage tables even though it measures distance to the real distribution. A mode-collapsed model that reproduces training layouts scores well; the BCA/DS paper shows FID keeps improving while per-condition diversity collapses. (v) It is blind to overlaps, disconnected rooms or missing doors as long as the colour statistics match.

### 2.5 KID / LPIPS-COV / MMD / Diversity Score. Category: distribution and diversity
- **KID** (GSDiff, FloorplanSBS, FloorplanGSO, GRE-Diff) is the unbiased squared MMD with a polynomial kernel on Inception features. GSDiff calls it "more robust than FID", and GRE-Diff writes out the formula. It has the same rendering caveats as FID.
- **COV / MMD (LPIPS)** (GRE-Diff, FloorplanSBS MMD): GRE-Diff follows GET3D, generating |S_g| = 5|S_r| samples with LPIPS distance D. It defines COV = |{argmin_{x∈S_r} D(x,y) : y∈S_g}| / |S_r| and MMD = (1/|S_r|) Σ_{x∈S_r} min_{y∈S_g} D(x,y). COV measures mode coverage and MMD measures fidelity.
- **Diversity Score (DS)** (Stoppani et al., ESANN 2026, [arXiv 2602.01949](https://arxiv.org/abs/2602.01949)): DS = Tr(Cov({f(x_i)})) over Inception-V3 features of samples generated for a **fixed** input condition, computed from 100 samples × 4 conditions. The authors show that FID and boundary compatibility keep improving with training while DS collapses, i.e. "diversity collapse undiagnosed by FID".
- **iPLAN FID_area / FID_type** ([CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/He_iPLAN_Interactive_and_Procedural_Layout_Planning_CVPR_2022_paper.pdf)): FID computed on 1×K vectors (per-type average room area, or per-type room count) instead of Inception features. This is a cheap, render-free distribution metric.
- **GSDiff room-count statistics**: for each room type, the mean count in generated plans divided by the mean count in the dataset ("the closer to 1, the better").
- **Weaknesses:** LPIPS and Inception features still depend on the rendering. DS rewards noise unless it is paired with a validity metric. The ratio statistics are first-moment only.

### 2.6 Compatibility = graph edit distance. Category: condition compatibility
- **House-GAN**: "The compatibility with the bubble diagram is a graph editing distance between the input bubble diagram and the bubble diagram constructed from the output layout in the same way as the GT preparation". In House-GAN, two rooms are connected if the Manhattan distance between their bounding boxes is < 8 px. The paper cites Abu-Aisheh et al. 2015 (exact GED). It uses X = 1 sample per input because GED is expensive.
- **House-GAN++ / HouseDiffusion**: "modified GED" that accounts for doors (rooms are connected through door masks; details are in the House-GAN++ supplement). HouseDiffusion uses 1,000 samples × 5 rounds. Panoptic-Refine: "a door must be correctly placed between two rooms".
- **DStruct2Design / RLVR**: GED between the input bubble diagram and the diagram extracted from the output JSON. RLVR: "Compatibility can be interpreted as the number of connectivity mistakes". **Caveat:** RLVR reports best-of-10 samples, **selected by smallest overlap and then lowest Compatibility**, so the reported Compatibility is the minimum of 10 draws chosen using the metric itself.
- **GSDiff, FMLM, BCA/DS ("GC")**: same GED, following HouseDiffusion.
- **CE2EPlan, HouseMind**: GED between the **generated and GT** graphs (HouseMind uses NetworkX `graph_edit_distance`). This is a different quantity from compatibility with the input. HouseMind also reports **Node F1** (room-type set P/R) and **Edge Overlap** = |E_pred ∩ E_gt| / |E_pred ∪ E_gt|.
- **MSD** (ECCV 2024, [arXiv 2407.10121](https://arxiv.org/abs/2407.10121)): GED is replaced because doors are not predicted. Compatibility(Q;K) = (1/N) Σ_i (1/|E^k_i|) Σ_{e∈E^k_i} 1[e ∈ E^q_i], i.e. the fraction of target edges retained, where an edge exists if the minimum distance between areas is ≤ a buffer. Higher is better.
- **Weaknesses:** the score depends heavily on the graph-extraction heuristic (pixel thresholds, door detection). Scale grows with room count. Recall-only variants (MSD) can be gamed by adding spurious adjacencies. Plain GED doesn't distinguish a missing room from a missing door. Best-of-N selection on GED inflates scores. GED is NP-hard, so implementations use timeouts or approximations that are rarely stated.

### 2.7 IoU-type fidelity to GT. Category: condition fidelity (GT-dependent)
- **Graph2Plan**: average IoU between predicted room boxes and GT boxes (0.65 for the full model).
- **Tell2Design**: Micro IoU = Σ_r I_r / Σ_r U_r and Macro IoU = (1/R) Σ_r I_r/U_r, where I_r and U_r are the pixel intersection and union of GT and predicted rooms of type r. For image baselines the IoU is maximized over shifts of the plan centre. The authors themselves warn: "the same language instruction [can] map to multiple floor plan designs. Therefore, a low IoU score does not necessarily mean a bad generation."
- **ChatHouseDiffusion, HouseTune, HouseMind, GreenPlanner**: same Micro/Macro IoU. HouseMind adds **SSIM/PSNR**, and FP-LLaMa also reports SSIM/PSNR.
- **MSD**: MIoU(Q;K) = (1/N) Σ_i Σ_{c∈C} |R_c(Q_i) ∩ R_c(K_i)| / |R_c(Q_i) ∪ R_c(K_i)|. Polygons are drawn largest-first so small rooms are not occluded. The MSD challenge report (Kuhn 2023) reports IoU with background, without background and structure-only, as on the CodaLab leaderboard.
- **MaskPLAN**: MSE between generated and GT attribute vectors (per-type room count T, adjacency counts A, per-type summed area S, with area scaled by (20/256)²).
- **Weaknesses:** these metrics need paired GT and penalize valid alternative designs. They reward mode-averaging, so a deterministic model beats a diverse one; HouseTune notes Tell2Design's high IoU but low diversity. They are also sensitive to small shifts: ChatHouseDiffusion notes that small rooms can have IoU = 0 despite being near-correct.

### 2.8 Boundary / room constraint satisfaction. Category: constraint satisfaction (GT-free)
- **FMLM Boundary IoU**: "IoU to evaluate the distance between internal regions of boundaries and union of generated room regions".
- **GRE-Diff BC** (attributed to Cons2Plan, ACM MM 2024, and MaskPLAN; the Cons2Plan full text was not obtained): BC = mean over samples of [I_interior ∧ I_fill] × 100%. Here I_interior = no room pixel lies outside the boundary mask, and I_fill = no k×k (k = 10) window inside the boundary is left uncovered by rooms.
- **GRE-Diff RC**: the percentage of samples whose per-type room counts exactly match the requested counts. **F1_BC-RC** is the harmonic mean of BC and RC.
- **BCA/DS** uses a "Boundary Compatibility" from Cons2Plan.
- **Weaknesses:** these are binary and threshold-based. They don't check that rooms are usable shapes, and they can be satisfied by a single room filling the boundary unless combined with RC.

### 2.9 Geometric / functional validity. Category: geometric validity
- **Overlap** (DStruct2Design: "boolean check for existence of polygon overlap"). **RLVR** reports Overlap (boolean any-overlap) and **% Overlap** = total overlapped area / generated total area; overlapping or unparseable samples get reward 0 during RL. **GRE-Diff** reports the **overlap-free rate** = % of layouts with no positive-area intersection between any pair of room polygons.
- **Reachability rate** (GRE-Diff): % of layouts whose room graph, with rooms as nodes and derived door connections as edges, forms a single connected component.
- **FloorplanGSO Functionality / Connectivity** (EG 2026): Functionality = (number of the 4 basic types {living, bedroom, kitchen, bathroom} present) / 4. Connectivity = (number of "possible doors" under rules: living room ↔ entrance; each room sharing a wall with the living room has one door; balconies and bathrooms adjacent only to a bedroom have one door) / number of rooms.
- **GSDiff ablation**: Fake Edge % (misclassified wall edges) and Alignment Error (node alignment error).
- **GreenPlanner**: "Functional Rationality" = proportion meeting fire-safety distance (≤ 15 m walk from entrance), area range and connectivity, plus **EUI** (energy use intensity). Both are predicted by a learned ResNet-50 surrogate (PDE) trained on EnergyPlus labels.
- **Weaknesses:** the hand-designed rules are dataset- and culture-specific (RPLAN-centric). Surrogate-predicted metrics share biases with the generator when the same surrogate is used as the RL reward, as in GreenPlanner.

### 2.10 LLM-specific metrics
- **DStruct2Design** ([arXiv 2407.15723](https://arxiv.org/abs/2407.15723)) defines self-consistency and prompt-consistency.
  - *Self-consistency* checks: % difference between stated total area and the sum of stated room areas; % difference between a room's stated area and its polygon area; overlap (boolean); duplicate room IDs (boolean); stated `room_count` equal to the actual count; and room height/width from the polygon vs stated.
  - *Prompt-consistency* checks: % difference in number of rooms, total area and room area vs requested; precision/recall of room IDs and room types vs the prompt; and ID-type match.
- **RLVR**: Room Area = MAPE of per-room area vs prompt; Room ID = exact-match accuracy of IDs such as "bedroom|1". Invalid JSON is handled only as a zero reward during training and is **not reported** as a parse/validity rate.
- **ChatHouseDiffusion**: accuracy of the LLM (Llama3 / GPT-4-turbo / moonshot) in extracting room **type / size / location** from text into JSON (e.g. GPT-4-turbo 93.9 / 43.4 / 53.9).
- **HouseMind** (understanding tasks): Success rate = N_success / N_total, Room Match Rate, location accuracy, area difference (m²), adjacency accuracy and relation accuracy. Editing: ΔIoU of the changed region and Diff-MSE.
- **FP-LLaMa**: **ARS (Architectural Reasonableness Score)** is the output of FloorPlan-MPS, a preference model trained on ArchiMetricsNet (expert scores for functionality, flow and overall; 32 experts recruited, 29 kept). It acts as a learned judge and is also the RLHF reward.
- **Weaknesses:** self-consistency checks only measure internal bookkeeping, so a plan can be perfectly consistent and still unusable. Learned judges used as both reward and metric can be gamed (reward hacking). No paper found reports a standardized JSON parse/validity rate, so the denominators for the consistency metrics (are invalid outputs dropped?) are often unclear.

### 2.11 FloorplanGAN (Luo & Huang 2022): UNVERIFIED
The full text was not accessible (Elsevier, paywalled). Secondary sources only describe the architecture (vector generator + raster discriminator). Its metrics are not reported here.

---

## 3. The "realism" answer

Floor-plan papers that report a metric explicitly named **Realism**, and how each computes it:

| Paper | Venue / year | Computation |
|---|---|---|
| **House-GAN** (Nauata et al.) | ECCV 2020 | Average user rating. Generated layout vs GT or vs another method; ratings better +1, worse −1, equally-good +1, equally-bad −1. 12 grad students + 10 architects, 75 pairs each. One "All groups" number plus pairwise difference matrices. **The origin of the metric.** |
| **House-GAN++** | CVPR 2021 | "Which is more realistic or a tie": win +1 / loss −1 / tie 0. 100 pairs per method pair, 10 amateurs + 10 architects, on raw masks and vectorized plans. The average of pairwise scores is the realism metric. |
| **Panoptic-Refine** (Liu et al.) | ECCV 2022 | Ranking of 3 methods per bubble diagram (100 diagrams) by 10 architecture-trained + 7 amateur respondents on "visual quality and functionality": +1 for 1st, 0 for 2nd, −1 for 3rd. |
| **HouseDiffusion** | CVPR 2023 | A/B/equal, +1/−1/0. 150 evaluations per system pair by 10 participants. **Reported vs GT only** (−0.19). Omitted for Non-Manhattan. |
| **GTGAN** | CVPR 2023 | House-GAN protocol (12 PhD students + 10 architects, 75 comparisons, 4-level rating). |
| **HouseLLM / HouseTune** | arXiv 2024–25 | "Same procedure as House-GAN++". Score −0.03; "65% … comparable to the ground truth". |
| **Floor-plan RLVR** (Lara et al.) | Findings of ACL 2026 | 40 volunteers × 10 randomized GT-vs-generated pairs; +1 generated chosen, −1 GT chosen, 0 equal; mean over all. Score 0.028. |
| **GRE-Diff** | 2026 (arXiv 2607.08086) | Section titled "Realism": 58 CS grad students, 1,740 forced-choice responses (GT/Ours/Cannot tell; Graph2Plan/Ours/Cannot tell). Reported as % preferring ours (60.69% vs GT). |

Papers that use the word "realism" for something that is **not** a user-rated Realism score:
- **WallPlan** (SIGGRAPH 2022): its study measures "generation quality and perceived realism", but the question asked is "more plausible", reported as % chosen.
- **BCA/DS** (ESANN 2026, *"Balancing Realism and Diversity"*): "realism" means FID.
- **GreenPlanner**: FID is called "visual realism".
- **HouseMind**: FID and GED are called "distributional realism".
- **MSD** (ECCV 2024): "Realism" is a dataset-description subsection about how realistic RPLAN/LIFULL/MSD data are, not a metric.

**Most likely candidates for "the paper with a realism metric":** a CVPR/ECCV paper with a Realism / Diversity / Compatibility table points to **House-GAN (ECCV 2020)**, **House-GAN++ (CVPR 2021)** or **HouseDiffusion (CVPR 2023)**. If it was an LLM paper, it is probably **Floor-plan RLVR (Findings ACL 2026)** or **HouseTune/HouseLLM**. If "realism" was the title of a user-study section in a 2026 diffusion paper, it is probably **GRE-Diff**.

---

## 4. Trends

**What fell out of use**
- **The 4-level House-GAN rating** (better/equally-good/worse/equally-bad) was replaced by 3-way win/tie/loss (House-GAN++). HouseDiffusion then switched to reporting only vs GT, and that has become the norm (RLVR 2026).
- **Realism user studies in the House-GAN style are declining.** Many 2024–2026 diffusion papers (GSDiff, MaskPLAN, FloorplanSBS, FloorplanGSO, CE2EPlan tables) rely on FID/KID plus an optional small forced-choice study, and baselines' realism is copied from HouseDiffusion Table 1.
- **Box IoU vs GT** (Graph2Plan) and **group-wise k-fold FID** (House-GAN's 5 groups) gave way to the 5/6/7/8-room RPLAN protocol.
- Calling FID "Diversity" persists only in the House-GAN lineage (through RLVR 2026). Elsewhere FID is called realism/fidelity, and diversity is measured separately (DS, COV).

**What became standard**
- FID is near-universal, and KID is common since 2024 (GSDiff onward). GED-to-input "Compatibility" is standard for any bubble-diagram-conditioned method, including LLM methods.
- Since 2024 there has been a shift to **GT-free validity metrics**: overlap rate or % overlap, overlap-free rate, reachability, boundary constraint (BC), room constraint (RC), functionality and connectivity. These metrics are verifiable and can double as RL rewards.

**New LLM-era metrics (2024–2026)**
- **Self- and prompt-consistency** of structured JSON outputs (DStruct2Design): area MAPE, room-count/ID/type precision and recall, stated-vs-polygon area.
- **Verifiable reward metrics used as both RL reward and evaluation** (RLVR: connectivity, total-area error, overlap as a hard constraint). This raises the risk that the metric gets optimized directly, especially with best-of-N selection on the evaluation metric itself.
- **LLM parsing accuracy** of text → JSON specs (ChatHouseDiffusion), and **understanding / editing metrics** (HouseMind success rate, RMR, ΔIoU; GRE-Diff Tiny-ROE).
- **Learned preference judges** (FP-LLaMa's FloorPlan-MPS / ARS), trained on expert scores and used for RLHF and evaluation.
- **Domain-performance surrogates** (GreenPlanner EUI and fire-safety rationality via a learned PDE).
- **Critiques of FID**: BCA/DS (2026) shows FID and constraint scores improving while per-condition diversity collapses. FP-LLaMa argues that FID, GED and PSNR ignore architectural domain knowledge.

**Gaps that remain (useful for `fpeval`)**
- There is no shared renderer for FID: palette, resolution (256 vs 512), and whether walls and doors are drawn all differ by paper.
- Inter-rater agreement is almost never reported; DiffPlanner's Fleiss' κ = 0.1–0.2 is the exception.
- No paper reports a standardized parse/validity rate with clear denominators for LLM outputs.
- Building-code or norm compliance (minimum room sizes, daylight, egress) is essentially absent, apart from GreenPlanner's fire-distance and area checks via a surrogate.
