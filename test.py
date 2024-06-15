import gradio

css = """
    .diffusion-spave-div h1 {
        font-size: 3rem;
        font-weight: bold;
    }
    .finetuned-diffusion-div div {
        display: inline-flex;
        align-items: center;
        gap: .8rem;
        font-size: 1.75rem
    }
    .finetuned-diffusion-div div h1 {
        font-weight: 900;
        margin-bottom: 7px;
    }
    .finetuned-diffusion-div p{
        margin-bottom: 10px;
        font-size: 94%;
    }
    a {
        text-decoration:underline
    }
    .tabs{
        margin-top: 0;
        margin-bottom: 0;
    }
    #gallery{
        min-height: 20rem
    }
    .custom-button {
        --width: 200px;
        --height: 35px;
        --tooltip-height: 35px;
        --tooltip-width: 90px;
        --gap-between-tooltip-to-button: 18px;
        --button-color: #1163ff;
        --tooltip-color: #fff;
        width: var(--width);
        height: var(--height);
        background: var(--button-color);
        position: relative;
        text-align: center;
        border-radius: 0.45em;
        font-family: "Arial";
        transition: background 0.3s;
        font-size: 12;
        text-decoration: none;
    }
    .custom-button:hover {
        background: #6c18ff;
    }
    #prompt-space {
        opacity: 1;
    }

"""

with gradio.Blocks(css=css) as app:
    gradio.HTML(
         f"""
            <div class="diffusion-spave-div">
              <div>
                <h1>Diffusion Space</h1>
              </div>
            </div>
        """
    )
    with gradio.Row():
        with gradio.Column(scale=20):
            with gradio.Tab("Prompting"):
                generate = gradio.Button(value="Documentation", variant="secondary", elem_classes="custom-button",
                                         link="https://sygil-dev.github.io/sygil-webui/docs/Installation/docker-guide/")
                
        with gradio.Column(scale=50):
            with gradio.Group():
                gallery = gradio.Gallery(
                    label="Generated image",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    container=True,
                    interactive=True,
                    show_share_button=True,
                    show_download_button=True,
                    object_fit="fill",
                    min_width=600,
                )
                
            settings = gradio.Markdown()
            error_output = gradio.Markdown()
            
        with gradio.Column(scale=30):
            with gradio.Tab("Controllers"):
                generate = gradio.Button(value="Generate", variant="secondary")
            
            gradio.HTML(
                f"""
                    <div class="normal-text">
                    <div>
                        <p>Try to prompt in the most detailed way for the deep learning model
                        to produce the best result. Feel free to try as many prompt as you can.</p>
                    </div>
                    </div>
                """
            )
            
            with gradio.Group():
                prompt = gradio.Textbox(label="Prompt", show_label=False, 
                                        max_lines=3, placeholder="Enter prompting text", 
                                        lines=10, container=False, elem_id="prompt-space")
                
                neg_prompt = gradio.Textbox(label="Negative prompt", show_label=True, 
                                            placeholder="What to exclude from the image")
                
                with gradio.Row():
                    n_images = gradio.Slider(label="Images", value=1, minimum=1, maximum=12, step=1)
                    seed = gradio.Slider(minimum=0, maximum=2147483647, label='Seed', value=0, step=1)

                with gradio.Row():
                    guidance = gradio.Slider(label="Guidance scale", value=7, maximum=20, step=1)
                    steps = gradio.Slider(label="Steps", value=20, minimum=2, maximum=50, step=1)

                with gradio.Row():
                    width = gradio.Slider(label="Width", value=768, minimum=64, maximum=1920, step=64)
                    height = gradio.Slider(label="Height", value=768, minimum=64, maximum=1920, step=64)

                scheduler_dd = gradio.Radio(label="Scheduler", value="euler_a", type="value",
                                            choices=["euler_a", "euler", "dpm++", "ddim", "ddpm", "pndm", "lms", "heun", "dpm"])
            

app.launch(debug=True)