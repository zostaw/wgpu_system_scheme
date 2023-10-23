import numpy as np


def generate_circle(steps=180, radius=0.2, file="circles.rs", center=[-0.03, -0.45], color=[0.002, 0.002, 0.002], depth=0.0):
    PI = np.pi
    indices = np.arange(steps)
    angles = indices / steps * 2 * PI

    xes = radius * np.cos(angles)
    yes = radius * np.sin(angles)

    with open(file, "w") as file:
        if steps < 3:
            raise("fProvided {steps} steps, but should be at least 3. Program will generate output, but 3 steps is too little to look like circle, "
                  "consider big number from which the circle will be built).")
            exit(1)
        # start with center
        file.write("use crate::Vertex;\n")
        file.write("pub fn circles() -> (Box<[Vertex]>, Box<[u16]>) {\n\n")
        file.write(f"    let vertices: Box<[Vertex]> = Box::new([\n")
        file.write("        Vertex {\n")
        file.write(f"            position: [{center[0]}, {center[1]}, {depth}],\n")
        file.write(f"            color: [{color[0]}, {color[1]}, {color[2]}],\n")
        file.write("        },\n")

        for x, y in zip(xes, yes):
            file.write("        Vertex {\n")
            file.write(f"            position: [{center[0] + x}, {center[1] + y}, {depth}],\n")
            file.write(f"            color: [{color[0]}, {color[1]}, {color[2]}],\n")
            file.write("        },\n")
        file.write("    ]);\n\n")

        # indices
        file.write(f"    let indices: Box<[u16]> = Box::new([\n")
        for index in range(1, steps-1):
            file.write(f"        0, {index}, {index + 1},\n")

        # last vertex connects back to 1st
        file.write(f"        0, {steps}, 1,\n")

        file.write("    ]);\n\n")
        file.write("    return (vertices, indices);\n")
        file.write("}\n")

if __name__ == "__main__":
    generate_circle()
