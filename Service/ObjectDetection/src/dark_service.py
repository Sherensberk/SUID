from core.core import service                
from codetiming import Timer

from interface.dark_interface import Net
@Timer(name="DarkHelpInterfaceProcess",  text="{name} demorou: {:.4f} segundos")
def callback(command, darknet):
    action = command.get('action')
    match action:
        case 'image': # {'action': image, value:{'original_image': <byte_array>, 'info':<dict>}}
            image = command['value']['original_image']
            
            if image is not None:
                x = darknet.predict(image)
                command |= {"darknet_analisis":x}
                command['action'] = 'interpret'
                command['info']['from'] = 'dark'
                command['to_output']= x is not False
                print(x)
                return command
        case _:
            pass
    return {}

if __name__ == "__main__":
    import argparse
    from threading import Thread

    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--service_host', type=str, required=False, help="(HOST) where the this service are.", default="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="5000")
    parser.add_argument('--output_host', type=str, required=False, help="(HOST) where the information goes.", default="yolo")
    parser.add_argument('--output_port', type=str, required=False, help="(PORT) where the information goes.", default="6000")
    parser.add_argument('--cfg', type=str, required=False, help="(cfg) filepath.", default="6000")
    parser.add_argument('--names', type=str, required=False, help="(names) filepath.", default="6000")
    parser.add_argument('--weights', type=str, required=False, help="(weights) filepath.", default="6000")
    
    args = parser.parse_args()

    # dh = DarkHelp.CreateDarkHelpNN((args.cfg).encode("utf-8"), (args.names).encode("utf-8"), (args.weights).encode("utf-8"))
    md =  Net((args.cfg).encode("utf-8"), (args.names).encode("utf-8"), (args.weights).encode("utf-8"))
    callback_args=(md,)

    server_process = Thread(target=service, args=((args.service_host, int(args.service_port)), ("interpreter", int(args.output_port)), callback, *callback_args))

    print(f"{__file__} started.")
    server_process.start()
    server_process.join()
    print(f"{__file__} stopped.")
    