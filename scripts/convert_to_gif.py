import imageio
import os
import sys

def convert_mp4_to_gif(input_path, output_path, fps=30, resize_factor=0.5):
    """
    Converts an mp4 video to a GIF.
    
    Args:
        input_path (str): Path to the mp4 file.
        output_path (str): Path to save the gif.
        fps (int): Frames per second.
        resize_factor (float): Scale factor to keep the GIF size small (0.5 = 50% size).
    """
    print(f"Converting {input_path} to {output_path}...")
    
    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    
    writer = imageio.get_writer(output_path, fps=fps, loop=0)
    
    for frame in reader:
        # Optional: Resize here if the gif is too large, 
        # but imageio doesn't have a simple built-in resize. 
        # We'll stick to full resolution for now.
        writer.append_data(frame)
        
    writer.close()
    print("Done!")

if __name__ == "__main__":
    in_file = "docs/videos/agent_performance_1750.mp4"
    out_file = "docs/figures/agent_demo.gif"
    
    if os.path.exists(in_file):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        convert_mp4_to_gif(in_file, out_file)
    else:
        print(f"Error: {in_file} not found. Run the video agent script first.")
