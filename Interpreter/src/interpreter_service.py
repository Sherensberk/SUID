from core import service
from threading import Thread
from interface.interpreter_interface import Item, Plane
from codetiming import Timer
import cv2

@Timer(name="InterpretPredctionsProcess",  text="{name} demorou: {:.4f} segundos")
def callback(command, plane:Plane):
    action = command.get('action')

    match action:
        case 'interpret':
            match command['info']['from']:
                case 'dark':
                    i = command['value' ]['original_image'].copy()
                    plane.update_real_image(i)
                    
                    centers = []
                    #? Item() deveria ser enviado junto a mensagem.
                    for f in command['darknet_analisis']['file']:
                        for p in f['prediction']:
                            centers.append(Item(
                                f"({p['prediction_index']}) {p['name']}",
                                {
                                    'x':p['original_point']['x']*f['original_width'],
                                    'y':p['original_point']['y']*f['original_height'],
                                },
                                plane=plane
                            ).coordinate.real)
                    frame = cv2.undistort(i, RI, DI, None, URI)
                    b = cv2.fillPoly(frame.copy(), plane.virtual_plane, (0,0,0))
                    frame = cv2.addWeighted(b, 0.4, frame, 1-0.4, 0)
                    print(centers)
                    print(plane.virtual_plane)
                    print(plane.virtual_plane_size)
                    cv2.imwrite('/src/vw.jpg', plane.mask)
                    cv2.imwrite('/src/vp.jpg', frame)
                    return {}

if __name__ == "__main__":
    import argparse
    import cv2.aruco as aruco
    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--service_host', type=str, required=False, help="(HOST) where the this service are.", default="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="5000")
    parser.add_argument('--output_host', type=str, required=False, help="(HOST) where the information goes.", default="tensorflow")
    parser.add_argument('--output_port', type=str, required=False, help="(PORT) where the information goes.", default="6000")
    
    parser.add_argument('--working_plane', action='store_true')
    parser.add_argument('--marker_type', type=str, required=False )
    parser.add_argument('--board_shape_width', type=str, required=False)
    parser.add_argument('--board_shape_height',type=str, required=False)
    parser.add_argument('--square_size', type=str, required=False)
    parser.add_argument('--marker_size', type=str, required=False) 
    parser.add_argument('--id_offset', type=str, required=False) 
    
    args = parser.parse_args()
    plane = Plane(
        getattr(aruco, args.marker_type),
        (int(args.board_shape_width), int(args.board_shape_height)),
        float(args.square_size),
        float(args.marker_size),
        int(args.id_offset)
    )
    RI, URI, DI, ROI = plane.calibrate_from_dir('./calibration/charuco/')

    callback_args = (plane,)
    server_process = Thread(target=service, args=((args.service_host, int(args.service_port)), (args.output_host, int(args.output_port)), callback, *callback_args))

    print(f"{__file__} started.")
    server_process.start()
    server_process.join()
    print(f"{__file__} stopped.")