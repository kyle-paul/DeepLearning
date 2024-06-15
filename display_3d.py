import gradio as gr

def load_mesh(mesh_file_name):
    return mesh_file_name

app = gr.Interface(
    fn=load_mesh,
    inputs=gr.Model3D(),
    outputs=gr.Model3D(
        clear_color=[0.0, 0.0, 0.0, 0.0],
        label="3D Model",
    ),
)

if __name__ == "__main__":
    app.launch(debug=True, share=True)