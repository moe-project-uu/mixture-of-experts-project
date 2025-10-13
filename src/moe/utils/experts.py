"""keep all expert-building logic (like small MLPs or convolutional blocks) 
in one place so every MoE variant can reuse the same code instead of duplicating
 expert definitions. When we later tweak activation functions, layer sizes, or 
 add dropout, we only change it once, and all heads stay consistent."""