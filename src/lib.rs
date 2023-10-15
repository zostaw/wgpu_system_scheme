use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder, dpi::PhysicalPosition,
};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

// lib.rs
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }

    fn change_position(&mut self, new_pos: [f32; 3]) {
        self.position[0] = self.position[0] + new_pos[0];
        self.position[1] = self.position[1] + new_pos[1];
        println!("position[0] {}, position[1] {}", self.position[0], self.position[1]);
    }
}

#[derive(Debug)]
struct VisualObject {
    vertices: Box<[Vertex]>,
    indices: Box<[u16]>,
}

impl VisualObject{
    fn new(vertices: Box<[Vertex]>, indices: Box<[u16]>) -> VisualObject {
        return VisualObject { vertices, indices };
    }
}

struct Rectangle {
    // Implements VisualObject
}

impl Rectangle{
    fn new(left_down_pos: [f32; 2], right_up_pos: [f32; 2], depth: f32, color: [f32; 3]) -> VisualObject {
        // Implementation takes args:
        // left_down_pos - left/down corner
        // right_up_pos - right/up corner
        // color
        // These are used to generate Rectangle vertices and indices.
        
        let vertices: [Vertex; 4] = [
            Vertex {
                position: [left_down_pos[0], left_down_pos[1], depth],
                color,
            }, // A
            Vertex {
                position: [right_up_pos[0], left_down_pos[1], depth],
                color,
            }, // B
            Vertex {
                position: [right_up_pos[0], right_up_pos[1], depth],
                color,
            }, // C
            Vertex {
                position: [left_down_pos[0], right_up_pos[1], depth],
                color,
            }, // D
        ];

        let indices: [u16; 5] = [0, 1, 2, 3, 0];

        VisualObject::new(Box::new(vertices), Box::new(indices))
    }
    
}

#[derive(Debug)]
struct Color {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}


struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    bg_color: Color,
    visual_objects: Vec<VisualObject>,
    vertex_buffers: Vec<wgpu::Buffer>,
    index_buffers: Vec<wgpu::Buffer>,
    index_buffers_indices: Vec<u32>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: Window,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let bg_color = Color {
            r: 0.01,
            g: 0.0147,
            b: 0.0114,
            a: 1.0,
        };

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",     // 1.
                buffers: &[Vertex::desc()], // 2.
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,                         // 2.
                mask: !0,                         // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
        });

        let vertex_buffers: Vec<wgpu::Buffer> = Vec::new();
        let index_buffers: Vec<wgpu::Buffer> = Vec::new();
        let index_buffers_indices: Vec<u32> = Vec::new();
        let visual_objects: Vec<VisualObject> = Vec::new();
        // let vertex_buffers: HashMap<&'a VisualObject, wgpu::Buffer> = HashMap::new();
        // let index_buffers: HashMap<&'a VisualObject, wgpu::Buffer> = HashMap::new();
        // let index_buffers_indices: HashMap<&'a VisualObject, u32> = HashMap::new();

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            bg_color,
            render_pipeline,
            visual_objects,
            vertex_buffers,
            index_buffers,
            index_buffers_indices,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    // impl State
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { 
                input:
                    KeyboardInput {
                        virtual_keycode: Some(VirtualKeyCode::Space),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                println!("some text");
                let object_vert_iter = self.visual_objects[0].vertices.iter_mut();
                for vertex in object_vert_iter {
                    vertex.change_position([0.1, -0.1, 0.0]);
                };
                self.reload_buffers();

            },
            _ => (),
        };
        false
    }

    fn reload_buffers(&mut self) {
        // TODO: I do not have consistency between buffers' id's and visual_objects id's - if one
        // of them breaks, the id's are out of sync
        // it must be rewritten
        for i in 0..self.vertex_buffers.len() {
            let vertices = &self.visual_objects[i].vertices;
            let indices = &self.visual_objects[i].indices;
            let device = &self.device;
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            self.vertex_buffers[i] = vertex_buffer;
            self.index_buffers[i] = index_buffer;
            self.index_buffers_indices[i] = indices.len() as u32;
        }

    }

    fn update(&mut self) {}

    fn append_visual_object(&mut self, visual_object: VisualObject) -> bool {
        // vertices: &[Vertex], indices: &[u16]
        let vertices = &visual_object.vertices;
        let indices = &visual_object.indices;
        let device = &self.device;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.vertex_buffers.push(vertex_buffer);
        self.index_buffers.push(index_buffer);
        self.index_buffers_indices.push(indices.len() as u32);
        self.visual_objects.push(visual_object);


        return false;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            b: self.bg_color.b,
                            g: self.bg_color.g,
                            r: self.bg_color.r,
                            a: self.bg_color.a,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);

            // for object in self.visual_objects {
            //     let vertex_buffer: wgpu::Buffer = self.vertex_buffers.entry(&object);
            //     render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            //     let index_buffer: wgpu::Buffer = self.index_buffers.entry(&object);
            //     render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            //     let index_buffers_indice: u32 = self.index_buffers_indices.entry(&object);
            //     render_pass.draw_indexed(0..index_buffers_indice, 0, 0..1);
            // }

            for i in 0..self.vertex_buffers.len() {
                    render_pass.set_vertex_buffer(0, self.vertex_buffers[i].slice(..));
                    render_pass.set_index_buffer(self.index_buffers[i].slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..self.index_buffers_indices[i], 0, 0..1);
            }
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}



fn initialize_visual_objects(state: &mut State) {

    let rectangle1 = Rectangle::new([-0.9, 0.4], [-0.3, 0.9], 0.0, [0.5, 0.0, 0.5]);
    let _ = &state.append_visual_object(rectangle1);

    let rectangle2 = Rectangle::new([-0.8, 0.2], [-0.2, 0.8], 0.0, [0.2, 0.2, 0.2]);
    let _ = &state.append_visual_object(rectangle2);

}

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // NOTE: this is for testing purpose
    window.set_outer_position(PhysicalPosition{x: 0.0, y: 0.0});

    let mut state = State::new(window).await;

    initialize_visual_objects(&mut state);


    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::CursorMoved { .. } => {
                    state.input(event);
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Space),
                            ..
                        },
                    ..
                } => {
                    state.input(event);
                }
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                }
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.window().request_redraw();
            }
            _ => {}
        }
    });
}
