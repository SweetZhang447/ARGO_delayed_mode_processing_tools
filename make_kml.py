"""
make_kml.py — Save data snapshot PNGs and generate a Google Earth KML file.

Output structure:
    save_dir/
        images/         ← one PNG per profile
        <float_name>.kml

The float name is derived from save_dir's folder name.

Run: python make_kml.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import simplekml
from tools import from_julian_day, read_intermediate_nc_file
from delayed_mode_processing import save_datasnapshot_graphs

def make_kml_with_snapshots(nc_filepath, save_dir, float_name):
    images_dir = save_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    float_data = read_intermediate_nc_file(nc_filepath)

    # Save one PNG per profile into images/
    save_datasnapshot_graphs(nc_filepath, images_dir)

    # Build KML
    kml = simplekml.Kml()
    trajectory_coords = []

    for i in np.arange(len(float_data["PROFILE_NUMS"])):
        prof_num = float_data["PROFILE_NUMS"][i]
        pnt = kml.newpoint(
            name=str(prof_num),
            coords=[(float_data["LONs"][i], float_data["LATs"][i])]
        )
        img_rel_path = Path("images") / f"profile_{prof_num}_data_snapshot.png"
        pnt.description = f"""
        <![CDATA[
            <img src="{img_rel_path.as_posix()}" width="600"/>
        ]]>
        """
        trajectory_coords.append((float_data["LONs"][i], float_data["LATs"][i]))

    line = kml.newlinestring(name=f"{float_name} Trajectory")
    line.coords = trajectory_coords
    line.style.linestyle.width = 3
    line.style.linestyle.color = simplekml.Color.black

    kml.save(str(save_dir / f"{float_name}.kml"))
    print(f"Saved {len(float_data['PROFILE_NUMS'])} snapshots and {float_name}.kml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Save profile snapshots and generate a Google Earth KML file.")
    parser.add_argument("-read_dir",   type=Path, required=True, help="Directory of intermediate netCDF files to read from.")
    parser.add_argument("-save_dir",   type=Path, required=True, help="Output directory. Images saved to save_dir/images/, KML saved to save_dir/<float_name>.kml.")
    parser.add_argument("-float_name", type=str,  required=True, help="Float name used for the KML filename and trajectory label (e.g. F9186).")
    args = parser.parse_args()

    nc_filepath = Path(r"c:\Users\szswe\Desktop\FLOAT_DATA\F9186\DMODE\F9186_VI")
    save_dir    = Path(r"c:\Users\szswe\Desktop\FLOAT_DATA\F9186\F9186_KML")
    float_name = "F9186"
    
    make_kml_with_snapshots(args.read_dir, args.save_dir, args.float_name)
