import os

from classes.plot import Plot
from classes.data import Data
from classes.sfmGlobal import SfmGlobal
from structured_from_motion_pair import StructedFromMotionPair


def StructedFromMotionSequential(SUPERIMAGEPAIRs):
    sfm = SfmGlobal(SUPERIMAGEPAIRs)
    camera_poses, points3d , points3d_color = sfm.run()

    return camera_poses, points3d, points3d_color


DATA = {
    'casa' : Data.get_casa,
    'chaleira' : Data.get_chaleira,
    'banana' : Data.get_banana,
    'banana2' : Data.get_banana2,
    'banana3': Data.get_banana3,
    'cubo': Data.get_cubo,
    'rosto': Data.get_rosto,
    'cachorro': Data.get_cachorro
}

if __name__ == '__main__':
    dataset = 'cubo'
    samples = 5
    paths = DATA[dataset](samples)
    print(paths)
    SUPERIMAGEPAIRs = []
    for en, (imag1Path, imag2Path) in enumerate(zip(paths[:-1], paths[1:])):
        superimagepair = StructedFromMotionPair(imag1Path, imag2Path,
                                                save_plot_dir=f"output/sfms_{dataset}_{samples}_{en}", show_plot=False)
        SUPERIMAGEPAIRs.append(superimagepair)

    camera_poses, points3d, points3d_color = StructedFromMotionSequential(SUPERIMAGEPAIRs)

    Plot.plot_cameras_frustum(
        camera_poses, points3d, points3d_color, points3d_size=15,
        save_path=os.path.join(f"output/sfms_{dataset}_{samples}", "07_3d_points_color.png"),
        show=True
    )

    Plot.show()