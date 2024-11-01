# gradiosaur
import gradio as gr

#image things
from PIL import Image,ImageDraw,ImageEnhance
import numpy as np
import io


#yolo
import ultralytics
from ultralytics import YOLO

#sam2
from sam2.build_sam import build_sam2_video_predictor,build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

#store keys
from dotenv import load_dotenv
import torch

#fastapi
#from fastapi import FastAPI

#foundation api
import openai
import anthropic

#tools for my app na
from tools import concat_images_with_labels,encode_image_to_base64,extract_frames_from_video,get_frame

#etc,paths,
import os
import pathlib
import dataclasses 
from dataclasses import dataclass,field
from hydra import initialize, compose
import hydra

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
my_api_key=os.getenv("anthropic_api_key")

client = anthropic.Anthropic(api_key=my_api_key)


#test sam2
def yolo_sam2(img_array):
    if isinstance(img_array, np.ndarray):
        img_p = io.BytesIO()
        Image.fromarray(img_array).save(img_p, format="PNG")
        img_p.seek(0)
        img = Image.open(img_p).convert("RGB")
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")


    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # with initialize(config_path="./sam2/checkpoints"):
    #     cfg = compose(config_name="sam2.1_hiera_t.yaml")
    #     sam2_model = build_sam2(cfg,"sam2.1_hiera_tiny.pt", device=device)
 
    sam2_model=build_sam2("./configs/sam2.1/sam2.1_hiera_t.yaml","/Users/kunkerdthaisong/preceptor/hack_midas_aigen/sam2/sam2/sam2.1_hiera_tiny.pt",device=device)
    sam_predictor = SAM2ImagePredictor(sam2_model)
    yolo_model = YOLO("/Users/kunkerdthaisong/preceptor/hack_midas_aigen/code/weights/yolo_best.pt").to(device=device)
    
    #img = Image.fromarray(img_p).convert("RGB")
    results = yolo_model(img)
    
    for result in results:
        boxes = result.boxes

        sam_predictor.set_image(img)
        
        for box in boxes:
            b = box.xyxy[0].tolist()  # Convert bounding box tensor to list [x_min, y_min, x_max, y_max]
            bboxes = torch.tensor([b])

            masks, scores, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bboxes,
                multimask_output=False,
            )


            draw = ImageDraw.Draw(img)
            draw.rectangle(b, outline="red", width=2)

            # Create a transparent mask overlay
            mask = masks[0].astype(bool)
            mask_img = Image.fromarray((mask * 255).astype("uint8")).convert("L")  # Convert to grayscale
            mask_overlay = Image.new("RGBA", img.size)
            mask_overlay.paste((0, 255, 0, 100), (0, 0), mask_img)  # Green overlay with alpha

            # Composite the original image and the mask overlay
            img = Image.alpha_composite(img.convert("RGBA"), mask_overlay)

    return img.convert("RGB")

def generate_medical_report(image_dict: dict) -> str:
    try:
        if image_dict is None or "composite" not in image_dict:
            return "Please upload an image and draw annotations first."

        composite_image = image_dict["composite"]

        base64_image = encode_image_to_base64(composite_image)

        prompt = """You are a endoscopist,colonoscopy and doctor.your job is to generate medical reports.

### **Imaging Findings:**
Provide a clear and structured description of any findings. Include anatomical details and describe abnormalities. you must provide all informantion.remember you must generate medical reports.
they can be the chance that an image was green or red highlight the polyp area.
**[Region/Organ System]:**
   - **Observation:** Describe the primary findings, including size, location, and shape of any abnormalities (e.g., lesions, masses, fractures,bleeding, fluid collections).
   - **Characteristics:** Describe key features such as margins, consistency (e.g., hypoechoic, hyperdense, heterogeneous), and any enhancement patterns.
   - **Location:** Specify the exact location, including proximity to or involvement of surrounding structures (e.g., infiltrating adjacent muscle or organs).
Ensure the report is clear and structured to be used by other healthcare providers in their clinical decision-making. do not provide time. you must provide all informantion.remember you must generate medical reports.
  """
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )

            return response.choices[0].message.content
        except:
            response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )
        return response.content[0].text

    except Exception as e:
        return f"Error generating report: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # SoS: Script of Scope 👨🏾‍⚕️🔬
        An Endoscopy-AI generate reports.
        """
    )

    frames = []
    frame_index = 0

    with gr.Row():
        video_input = gr.File(label="Upload Video", type="filepath",)
        frame_slider = gr.Slider(
            minimum=0,
            maximum=0,
            step=1,
            label="Select Frame",
            interactive=True
        )
        
    with gr.Row():
        mode_selector = gr.Radio(
            choices=["Edit Mode", "AI Prediction Mode"],
            label="Select Mode",
            value="Edit Mode"
        )

    with gr.Row():
        image_editor = gr.ImageEditor(
            type="numpy",
            label="Medical Image Editor",
            height=800,
            
        )
        image_preview = gr.Image(label="Preview with Annotations", height=800)

    generate_btn = gr.Button("Generate Medical Report", variant="primary")

    report_output = gr.Textbox(
        label="Generated Medical Report",
        lines=15,
        placeholder="Medical report will appear here...",
    )

    def load_video(file):
        global frames, frame_index
        frames = extract_frames_from_video(file.name)
        frame_index = 0
        return get_frame(frames, frame_index), gr.update(maximum=len(frames) - 1)

    def update_frame(index):
        global frame_index
        frame_index = int(index)
        return get_frame(frames, frame_index)
    
    
    def handle_image_update(image_data, mode):
        if image_data is not None and isinstance(image_data, dict) and "composite" in image_data:
            if mode == "AI Prediction Mode":
                return yolo_sam2(image_data["composite"])
            else:
                
                return image_data["composite"]
        return None


    video_input.change(
        load_video,
        inputs=video_input,
        outputs=[image_editor, frame_slider]
    )

    frame_slider.change(
        update_frame,
        inputs=frame_slider,
        outputs=image_editor
    )
    
    image_editor.change(
        handle_image_update,
        inputs=[image_editor,mode_selector],
        outputs=image_preview,
        show_progress="hidden",
    )
    
    generate_btn.click(
        generate_medical_report, inputs=[image_editor], outputs=[report_output]
    )



if __name__ == "__main__":
    demo.launch(share=True)
    #yolo_sam2("saved.png")