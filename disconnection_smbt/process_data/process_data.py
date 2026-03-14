import os
import subprocess
import sys

def run_script(script_path, env_name, change_dir=False):
    original_dir = os.getcwd()
    if change_dir:
        script_dir = os.path.dirname(os.path.abspath(script_path))
        os.chdir(script_dir)
        script_name = os.path.basename(script_path)
    else:
        script_name = script_path

    cmd = f"conda run -n {env_name} python {script_name}"
    
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"Running {script_path}...")
        
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            if output == '' and error == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
            if error:
                print(error.strip(), file=sys.stderr)
        
        return_code = process.poll()
        
        if return_code == 0:
            print(f"Successfully ran {script_path}")
        else:
            print(f"Error running {script_path}. Return code: {return_code}")
            sys.exit(1)
    
    finally:
        if change_dir:
            os.chdir(original_dir)

def process_data():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    steps = [
        ("generate_smi.py", "ml_py3.10", False),
        ("../bt_attention/generate_disconnect.py", "bt_emb", True),
        ("clean_disconnect.py", "ml_py3.10", False),
        ("generate_smiles_id_mapping.py", "ml_py3.10", False),
        ("generate_matrix.py", "ml_py3.10", False)
    ]
    
    for script, env, change_dir in steps:
        run_script(script, env, change_dir)
    
    print("All processing steps completed successfully.")

if __name__ == "__main__":
    process_data()
