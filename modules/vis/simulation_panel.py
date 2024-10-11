import viser

def add_gui_object_group(
    server: viser.ViserServer,
    obj_name: str,
    drop_vel: float = 0.0,
    weight: float = 1.0,
    neuma: str | None = None
):
    with server.gui.add_folder(obj_name):
        
        gui_drop_vel = server.gui.add_slider(
            "Drop Vel",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=drop_vel,
        )

        gui_weight = server.gui.add_slider(
            "Weight",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=weight,
        )

        gui_neuma = server.gui.add_dropdown(
            "NeuMA", ("bouncy", "clay", "honey", "jelly", "rubber", "sand"),
            initial_value=neuma,
        )

        gui_object_exists = server.gui.add_checkbox(
            "Object Exists", initial_value=True
        )

    return gui_drop_vel, gui_weight, gui_neuma, gui_object_exists