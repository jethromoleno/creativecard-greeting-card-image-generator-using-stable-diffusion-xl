<h1>🎨 CreativeCard — Greeting Card Image Generator (Stable Diffusion XL)</h1>

CreativeCard is an AI-powered greeting card generator built with Stable Diffusion XL (SDXL).
Users can generate high-quality greeting card images with custom prompts, add short messages, and export the final card easily.

This project demonstrates real-world Generative AI, UI design, and image processing skills.

Youtube Video Demo Link:  [![YouTube Video Demo](https://img.shields.io/badge/YouTube-Video-red?style=flat&logo=youtube)](https://www.youtube.com/watch?v=LSXt0PSaVdM)

Documentation: [![Documentation](https://img.shields.io/badge/documentation-link-brightgreen?style=flat)](https://github.com/jethromoleno/creativecard-greeting-card-image-generator-using-stable-diffusion-xl/blob/main/docs/CreativeCard_Documentation.pdf)

User Manual: [![Documentation](https://img.shields.io/badge/documentation-link-brightgreen?style=flat)](https://github.com/jethromoleno/creativecard-greeting-card-image-generator-using-stable-diffusion-xl/blob/main/docs/User%20Manual.pdf)


<h2>✨ Features</h2>

🧠 Powered by Stable Diffusion XL

🎨 Generates greeting card styles (birthday, wedding, anniversary, Christmas, etc.)

📄 Add custom text on the generated card

🖼️ Adjustable resolution (up to 1429 × 2000 px — user preference)

⚙️ Loading bar inside the generation frame (UX improvement)

🚫 Disable “Generate” button while image is generating

💾 Save image via button (consistent UI size)

🖥️ CustomTkinter GUI for modern UI feel

<h2>🧠 Tech Stack</h2>

- Stable Diffusion XL (stabilityai/stable-diffusion-2-1-base or SDXL 1.0)
- Python (3.10 recommended)
- diffusers
- CustomTkinter
- PIL (Pillow)
- pyTorch
- NumPy
- OS
- Fork
- Transformers
- ctypes
- threading

<h2>⚙️ How It Works</h2>
1. User enters a greeting card prompt

Example:
“A warm birthday greeting card with soft pastel colors and cute characters.”

2. SDXL pipeline processes the text input

Tokenizes the prompt
Runs through the UNet + VAE
Upscales automatically to your preferred 1429 × 2000 resolution

3. Generated image appears in the UI

Progress bar updates inside frame
"Generate" button is disabled while generating

4. User can save the generated card

Saved as a PNG or JPG.

<h2>👨‍💻 Author</h2>

- Jethro P. Moleno
- Computer Engineering – Mapúa University
- Email: jethromoleno@gmail.com
- LinkedIn: www.linkedin.com/in/jethromoleno
