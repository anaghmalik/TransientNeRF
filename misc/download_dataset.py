import subprocess
import configargparse
import os 

def download_dataset(scenes):
    link_dict = {
        "bench":"https://www.dropbox.com/scl/fo/a7c6ej2zj5julibmi0oe7/AOzgkS7_PkMNzg4ybZBTTZ8?rlkey=jjkqqz6ifi2ib3g01yf03j12h&st=v02sztb4&dl=0", 
        "lego":"https://www.dropbox.com/scl/fo/xrvnu5i7cwg6ng4iveerd/APXNbHRR2IJqP6q384ZrsTY?rlkey=qag95l1xn3hgebxqcqb4grdbw&st=1xjm5ob9&dl=0",
        "chair":"https://www.dropbox.com/scl/fo/rw1kx28apf2nj9nmmankn/AAXBD63yuU-TcaKJRh2WlUQ?rlkey=58tb99g1d41ml4asqxdp5lrxg&st=og6clu8k&dl=0",
        "ficus":"https://www.dropbox.com/scl/fo/ash43k5stxykgu82y0rty/AJM0R8PdduoE4BMl31BI7xY?rlkey=ad1uacytq2mr5e0hc4tpmm7xp&st=uwm356c4&dl=0", 
        "hotdog":"https://www.dropbox.com/scl/fo/lrbe9b8tsmpu6m3e25s7q/ANANvfFhEdoib9rHv5QmPwo?rlkey=bea0k5zi6ahgts88mlpvs4mta&st=8itg14zh&dl=0", 
        "boots":"https://www.dropbox.com/scl/fo/cx5tl37qjytcv8v652q00/ADOzXV5AbW0_y_wpptIk8PY?rlkey=ne75mf8kc3hg6wg7coak74ww5&st=hklnyi76&dl=0", 
        "carving":"https://www.dropbox.com/scl/fo/962tg8nqtx42m7lasqo7s/AIAl3BFPt7U1xD8eanwZnP8?rlkey=qm3r7d52dnvroac8lwsbq93pb&st=58hvm91y&dl=0", 
        "baskets":"https://www.dropbox.com/scl/fo/jxvcp63h0z7u0hptk2b0f/AOvcyi2RB0R-999nzQZtwvk?rlkey=ce229qfku8brdfznzl62fi792&st=yf4aii6e&dl=0", 
        "chef":"https://www.dropbox.com/scl/fo/15mcemsypabotbsr07gf0/AIdsXG0GHVp_FLf7i95Orr4?rlkey=abugupv7fo1gtjue73h0jm8uf&st=omb14j7c&dl=0",
        "cinema":"https://www.dropbox.com/scl/fo/hdytz9zw73jq2f3hxp44b/ACQvEjYApQ6hDfhpf2vvoSw?rlkey=caeizu4zklzziyzt3wlcb4pgg&st=nherw66u&dl=0",
        "food":"https://www.dropbox.com/scl/fo/brg2po03txm7nftnljhyb/AFfgHP1517nCEfY1oVGhfYo?rlkey=znxzp37wao2y9bydq8ceapimr&st=8kfsaiob&dl=0", 
        "intrinsics.npy":"https://www.dropbox.com/scl/fi/s4whgdajo5jy5wmdtlhwf/intrinsics.npy?rlkey=s247t4pvtubn726dzpnwmnlrv&st=0hnobsb9&dl=0", 
        "pulse_low_flux.mat":"https://www.dropbox.com/scl/fi/x63omsjecjijpg5q5dpiw/pulse_low_flux.mat?rlkey=1dm9d8jrdrogtw2xbgg2x8x17&st=4gll02on&dl=0", 
    }
    os.makedirs("dataset", exist_ok = True)
    for file in ["pulse_low_flux.mat", "intrinsics.npy"]:
        command = f'wget "{link_dict[file]}" -O ./dataset/{file}'
        subprocess.run(command, shell=True)
        

    for folder in scenes:
        os.makedirs(f"dataset/{folder}", exist_ok = True)

        command = f'wget "{link_dict[folder]}" -O dataset/{folder}.zip'
        subprocess.run(command, shell=True)
        
        command = f"unzip dataset/{folder}.zip -d dataset/{folder}"
        subprocess.run(command, shell=True)

        command = f"rm dataset/{folder}.zip"
        subprocess.run(command, shell=True)
    



if __name__=="__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', help='list of files to download')
    args = parser.parse_args()
    
    final_scenes = []
    all_scenes = ["cinema", "carving", "boots", "food", "chef", "baskets", "lego", "chair", "ficus", "hotdog", "bench"]
    
    if "all" in args.scenes:
        final_scenes = all_scenes.copy()
    
    if "captured" in args.scenes:
        scenes = ["cinema", "carving", "boots", "food", "chef", "baskets"]
    if "simulated" in args.scenes:
        scenes = ["lego", "chair", "ficus", "hotdog", "bench"]
        
    for scene in all_scenes:
        if scene in args.scenes:
            final_scenes += [scene]
    # final_scenes += ["pulse", "intrinsics"]

    final_scenes = list(set(final_scenes))
    download_dataset(final_scenes)



