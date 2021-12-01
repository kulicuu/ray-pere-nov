


use memoffset::offset_of;
use simple_logger::SimpleLogger;
use winit::{
    event::{
        Event, KeyboardInput, WindowEvent, 
        ElementState, StartCause, VirtualKeyCode,
        DeviceEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    window::Window         
};
// use winit::event::{ElementState, StartCause, VirtualKeyCode};
use structopt::StructOpt;

use erupt::{
    cstr,
    utils::{self, surface},
    vk, DeviceLoader, EntryLoader, InstanceLoader,
    vk::{Device, MemoryMapFlags},
};


use cgmath::{Deg, Matrix4, Point3, Vector3};


use std::{
    ffi::{c_void, CStr, CString},
    fs,
    fs::{write, OpenOptions},
    io::prelude::*,
    mem::*,
    os::raw::c_char,
    ptr,
    result::Result,
    result::Result::*,
    string::String,
    thread,
    time,
};

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};

const TITLE: &str = "Peregrine Ray-Trace";
const FRAMES_IN_FLIGHT: usize = 2;
const LAYER_KHRONOS_VALIDATION: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");


const SHADER_VERT: &[u8] = include_bytes!("../spv/s1.vert.spv");
const SHADER_FRAG: &[u8] = include_bytes!("../spv/s1.frag.spv");

#[derive(Debug, StructOpt)]
struct Opt {
    /// Use validation layers
    #[structopt(short, long)]
    validation_layers: bool,
}




unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {


    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open("./logs/log_main.txt")
        .unwrap();



  

    let data = CStr::from_ptr((*p_callback_data).p_message).to_string_lossy();
    // writeln!(file, &*data);
    file.write(&*data.as_bytes()).expect("Unable to write to file.");

    file.write(b"\n").expect("Unable to write newline to file.");
    


    eprintln!(
        "Vulkan: {}",
        CStr::from_ptr((*p_callback_data).p_message).to_string_lossy()
    );
    vk::FALSE
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct VertexV3 {
    pos: [f32; 4],
    color: [f32; 4],
}
impl VertexV3 {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 1] {
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Self, pos) as u32,
            },
            // vk::VertexInputAttributeDescription {
            //     binding: 0,
            //     location: 1,
            //     format: vk::Format::R32G32B32A32_SFLOAT,
            //     offset: offset_of!(Self, color) as u32,
            // },
        ]
    }
}


#[repr(C)]
#[derive(Clone, Debug, Copy)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}



fn main() {
    println!("Ray-trace Peregrine");


    // flush logs
    fs::remove_file("./logs/log_main.txt").unwrap();


    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Peregrine Ray-Trace")
        .build(&event_loop)
        .unwrap();

    let entry = EntryLoader::new().unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance
    let application_name = CString::new("Peregrine Ray-Trace").unwrap();
    
    let engine_name = CString::new("Vulkan Engine").unwrap();
    let app_info = vk::ApplicationInfoBuilder::new()
        .application_name(&application_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let mut instance_extensions = surface::enumerate_required_extensions(&window).unwrap();

    instance_extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);

    let mut instance_layers = Vec::new();
    instance_layers.push(LAYER_KHRONOS_VALIDATION);

    let device_extensions = vec![
        vk::KHR_SWAPCHAIN_EXTENSION_NAME, 
        vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        vk::KHR_RAY_QUERY_EXTENSION_NAME,
        vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        vk::KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        vk::KHR_SPIRV_1_4_EXTENSION_NAME,
        vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        vk::EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    ];

    let mut device_layers = Vec::new();

    device_layers.push(LAYER_KHRONOS_VALIDATION);

    let instance_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);

    let instance = unsafe { InstanceLoader::new(&entry, &instance_info, None) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers
    // if opt.validation_layers ...


    let messenger = {
        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT,
            )
            .pfn_user_callback(Some(debug_callback));

        unsafe { instance.create_debug_utils_messenger_ext(&messenger_info, None) }.unwrap()
    };


    let surface = unsafe { surface::create_surface(&instance, &window, None) }.unwrap();


    let (physical_device, queue_family, format, present_mode, device_properties) =
        unsafe { instance.enumerate_physical_devices(None) }
            .unwrap()
            .into_iter()
            .filter_map(|physical_device| unsafe {
                // println!("Physical Device: {:?}", physical_device);
                // println!("Phyisical Device Queue Family Properties: {:?}", instance.get_physical_device_properties(physical_device));
                let queue_family = match instance
                    .get_physical_device_queue_family_properties(physical_device, None)
                    .into_iter()
                    .enumerate()
                    .position(|(i, queue_family_properties)| {
                        queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                            && instance
                                .get_physical_device_surface_support_khr(
                                    physical_device,
                                    i as u32,
                                    surface,
                                )
                                .unwrap()
                    }) {
                    Some(queue_family) => queue_family as u32,
                    None => return None,
                };

                let formats = instance
                    .get_physical_device_surface_formats_khr(physical_device, surface, None)
                    .unwrap();
                let format = match formats
                    .iter()
                    .find(|surface_format| {
                        surface_format.format == vk::Format::B8G8R8A8_SRGB
                            && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
                    })
                    .or_else(|| formats.get(0))
                {
                    Some(surface_format) => *surface_format,
                    None => return None,
                };


                let present_mode = instance
                    .get_physical_device_surface_present_modes_khr(physical_device, surface, None)
                    .unwrap()
                    .into_iter()
                    .find(|present_mode| present_mode == &vk::PresentModeKHR::MAILBOX_KHR)
                    .unwrap_or(vk::PresentModeKHR::FIFO_KHR);

                let supported_device_extensions = instance
                    .enumerate_device_extension_properties(physical_device, None, None)
                    .unwrap();
                let device_extensions_supported =
                    device_extensions.iter().all(|device_extension| {
                        let device_extension = CStr::from_ptr(*device_extension);

                        supported_device_extensions.iter().any(|properties| {
                            CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                        })
                    });

                if !device_extensions_supported {
                    return None;
                }

                let device_properties = instance.get_physical_device_properties(physical_device);
                Some((
                    physical_device,
                    queue_family,
                    format,
                    present_mode,
                    device_properties,
                ))
            })
            .max_by_key(|(_, _, _, _, properties)| match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            })
            .expect("No suitable physical device found");
            //end of declaration of enum (physical_device, queue_family, format, present_mode, device_properties)



    println!("\n Using physical device: {:?} \n", unsafe {
        CStr::from_ptr(device_properties.device_name.as_ptr())
    });

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Logical_device_and_queues
    let queue_info = vec![vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];

    let features = vk::PhysicalDeviceFeaturesBuilder::new();

    let device_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_info)
        .enabled_features(&features)
        .enabled_extension_names(&device_extensions)
        .enabled_layer_names(&device_layers);

    let device =
        unsafe { DeviceLoader::new(&instance, physical_device, &device_info, None) }.unwrap();

    // let queue2 = unsafe { device2.get_device_queue(queue_family, 0) };
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
    let surface_caps =
        unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface) }
            .unwrap();
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }


    let (swapchain, swapchain_images, swapchain_image_views) = create_swapchain_etc(
        &surface,
        format,
        image_count,
        surface_caps,
        present_mode,
        &device
    );


    let entry_point = CString::new("main").unwrap();

    println!("\n \n");

    let model_path: &'static str = "assets/terrain__002__.obj";
    let (models, materials) = tobj::load_obj(&model_path, &tobj::LoadOptions::default()).expect("Failed to load model object!");
    let model = models[0].clone();
    let materials = materials.unwrap();
    let material = materials.clone().into_iter().nth(0).unwrap();
    let mut vertices = vec![];
    let mut indices = vec![];
    let mesh = model.mesh;
    let total_vertices_count = mesh.positions.len() / 3;
    for i in 0..total_vertices_count {
        let vertex = VertexV3 {
            pos: [
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
                1.0,
            ],
            color: [1.0, 1.0, 1.0, 1.0],
        };
        vertices.push(vertex);
    };
    indices = mesh.indices.clone(); 

    println!("Starting buffer and memory allocation/mapping processes... \n");


    let vertex_buffer_size = ::std::mem::size_of_val(&vertices) as vk::DeviceSize;
    
    println!("vertex_buffer_size: {:?}", vertex_buffer_size);

    let physical_device_memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    println!("\n physical_device_memory_properties: {:?}", physical_device_memory_properties);
    pretty_print(physical_device_memory_properties);


    let vertex_buffer_create_info = vk::BufferCreateInfoBuilder::new()
        .size(vertex_buffer_size * 200)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    println!("\n vertex_buffer_create_info: {:?}", vertex_buffer_create_info);

    let vertex_buffer = unsafe {
        device
            .create_buffer(&vertex_buffer_create_info, None)
            .expect("Failed to create vertex buffer.")
    };

    let vertex_buffer_memory_reqs = unsafe {
        device
            .get_buffer_memory_requirements(vertex_buffer)
    };
    println!("\n vertex_buffer_memory_reqs: {:?}", vertex_buffer_memory_reqs);

    let vertex_buffer_memory_allocate_info =
        vk::MemoryAllocateInfoBuilder::new()
                    .allocation_size(vertex_buffer_memory_reqs.size)
                    .memory_type_index(2);
                    // .build();
    println!("\n vertex_buffer_memory_allocate_info, {:?} \n", vertex_buffer_memory_allocate_info);

    let vertex_buffer_memory = unsafe {
        device
            .allocate_memory(&vertex_buffer_memory_allocate_info, None)
            .expect("Failed to allocate vertex buffer memory.")
    };
    println!("\n vertex_buffer_memory: {:?} \n", &vertex_buffer_memory);

    unsafe { device.bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0) }
        .expect("Error on bind buffer memory");


    unsafe {
        device
            .map_memory(
                vertex_buffer_memory,
                256,
                vk::WHOLE_SIZE,
                MemoryMapFlags::empty(),
            )
            .expect("Failed to map memory.");

    }


    // let uniform_transform = UniformBufferObject {
    //     model: Matrix4::from_angle_z(Deg(90.0)),
    //     view: Matrix4::look_at(
    //         Point3::new(2.0, 2.0, 2.0),
    //         Point3::new(0.0, 0.0, 0.0),
    //         Vector3::new(0.0, 0.0, 1.0),
    //     ),
    //     proj: {
    //         let mut proj = cgmath::perspective(
    //             Deg()
    //         )
    //     }

    // }





    let vert_decoded = utils::decode_spv(SHADER_VERT).unwrap();
    let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&vert_decoded);
    let shader_vert = unsafe { device.create_shader_module(&module_info, None) }.unwrap();

    let frag_decoded = utils::decode_spv(SHADER_FRAG).unwrap();
    let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&frag_decoded);
    let shader_frag = unsafe { device.create_shader_module(&module_info, None) }.unwrap();


    let shader_stages = vec![
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(shader_vert)
            .name(&entry_point),
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(shader_frag)
            .name(&entry_point),
    ];


    // let binding_description = VertexV3::get_binding_descriptions();
    // let attribute_description = VertexV3::get_attribute_descriptions();


    let binding_description = vk::VertexInputBindingDescriptionBuilder::new()
        .binding(0)
        .stride(std::mem::size_of::<VertexV3>() as u32,)
        .input_rate(vk::VertexInputRate::VERTEX);

    let binding_descriptions = &[binding_description][..];

    let attribute_description = vk::VertexInputAttributeDescriptionBuilder::new()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .offset(offset_of!(VertexV3, pos) as u32);

    let attribute_descriptions = &[attribute_description][..];


    let vertex_input = vk::PipelineVertexInputStateCreateInfoBuilder::new()
        .vertex_binding_descriptions(binding_descriptions)
        .vertex_attribute_descriptions(attribute_descriptions);



    let input_assembly = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);


    let viewports = vec![vk::ViewportBuilder::new()
        .x(0.0)
        .y(0.0)
        .width(surface_caps.current_extent.width as f32)
        .height(surface_caps.current_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)];
    let scissors = vec![vk::Rect2DBuilder::new()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(surface_caps.current_extent)];
    let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
        .viewports(&viewports)
        .scissors(&scissors);

    let rasterizer = vk::PipelineRasterizationStateCreateInfoBuilder::new()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_clamp_enable(false);

    let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlagBits::_1);

    let color_blend_attachments = vec![vk::PipelineColorBlendAttachmentStateBuilder::new()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)];
    let color_blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);

    let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new();
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
    let attachments = vec![vk::AttachmentDescriptionBuilder::new()
        .format(format.format)
        .samples(vk::SampleCountFlagBits::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

    let color_attachment_refs = vec![vk::AttachmentReferenceBuilder::new()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = vec![vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)];
    let dependencies = vec![vk::SubpassDependencyBuilder::new()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    let render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Conclusion
    let pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .unwrap()[0];

    // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Framebuffers
    let swapchain_framebuffers: Vec<_> = swapchain_image_views
        .iter()
        .map(|image_view| {
            let attachments = vec![*image_view];
            let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(surface_caps.current_extent.width)
                .height(surface_caps.current_extent.height)
                .layers(1);

            unsafe { device.create_framebuffer(&framebuffer_info, None) }.unwrap()
        })
        .collect();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers
    let command_pool_info =
        vk::CommandPoolCreateInfoBuilder::new().queue_family_index(queue_family);
    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }.unwrap();

    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swapchain_framebuffers.len() as _);
    let cmd_bufs = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }.unwrap();

    for (&cmd_buf, &framebuffer) in cmd_bufs.iter().zip(swapchain_framebuffers.iter()) {
        let cmd_buf_begin_info = vk::CommandBufferBeginInfoBuilder::new();
        unsafe { device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info) }.unwrap();

        let clear_values = vec![vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let render_pass_begin_info = vk::RenderPassBeginInfoBuilder::new()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: surface_caps.current_extent,
            })
            .clear_values(&clear_values);

        unsafe {
            device.cmd_begin_render_pass(
                cmd_buf,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, pipeline);


            device.cmd_bind_vertex_buffers(cmd_buf, 0, &[vertex_buffer], &[256]);


            device.cmd_draw(cmd_buf, indices.len() as u32, 1, 0, 0);
            device.cmd_end_render_pass(cmd_buf);

            device.end_command_buffer(cmd_buf).unwrap();
        }
    }

    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Rendering_and_presentation
    let semaphore_info = vk::SemaphoreCreateInfoBuilder::new();
    let image_available_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap())
        .collect();
    let render_finished_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap())
        .collect();

    let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight_fences: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_fence(&fence_info, None) }.unwrap())
        .collect();
    let mut images_in_flight: Vec<_> = swapchain_images.iter().map(|_| vk::Fence::null()).collect();








    // println!("\n \n");

    // let model_path: &'static str = "assets/terrain__002__.obj";
    // let (models, materials) = tobj::load_obj(&model_path, &tobj::LoadOptions::default()).expect("Failed to load model object!");
    // let model = models[0].clone();
    // let materials = materials.unwrap();
    // let material = materials.clone().into_iter().nth(0).unwrap();
    // let mut vertices = vec![];
    // let mut indices = vec![];
    // let mesh = model.mesh;
    // let total_vertices_count = mesh.positions.len() / 3;
    // for i in 0..total_vertices_count {
    //     let vertex = VertexV3 {
    //         pos: [
    //             mesh.positions[i * 3],
    //             mesh.positions[i * 3 + 1],
    //             mesh.positions[i * 3 + 2],
    //             1.0,
    //         ],
    //         color: [1.0, 1.0, 1.0, 1.0],
    //     };
    //     vertices.push(vertex);
    // };
    // indices = mesh.indices.clone(); 

    // println!("Starting buffer and memory allocation/mapping processes... \n");


    // let vertex_buffer_size = ::std::mem::size_of_val(&vertices) as vk::DeviceSize;
    
    // println!("vertex_buffer_size: {:?}", vertex_buffer_size);

    // let physical_device_memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    // println!("\n physical_device_memory_properties: {:?}", physical_device_memory_properties);
    // pretty_print(physical_device_memory_properties);


    // let vertex_buffer_create_info = vk::BufferCreateInfoBuilder::new()
    //     .size(vertex_buffer_size * 200)
    //     .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
    //     .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // println!("\n vertex_buffer_create_info: {:?}", vertex_buffer_create_info);

    // let vertex_buffer = unsafe {
    //     device
    //         .create_buffer(&vertex_buffer_create_info, None)
    //         .expect("Failed to create vertex buffer.")
    // };

    // let vertex_buffer_memory_reqs = unsafe {
    //     device
    //         .get_buffer_memory_requirements(vertex_buffer)
    // };
    // println!("\n vertex_buffer_memory_reqs: {:?}", vertex_buffer_memory_reqs);

    // let vertex_buffer_memory_allocate_info =
    //     vk::MemoryAllocateInfoBuilder::new()
    //                 .allocation_size(vertex_buffer_memory_reqs.size)
    //                 .memory_type_index(2)
    //                 .build();
    // println!("\n vertex_buffer_memory_allocate_info, {:?} \n", vertex_buffer_memory_allocate_info);

    // let vertex_buffer_memory = unsafe {
    //     device
    //         .allocate_memory(&vertex_buffer_memory_allocate_info, None)
    //         .expect("Failed to allocate vertex buffer memory.")
    // };
    // println!("\n vertex_buffer_memory: {:?} \n", &vertex_buffer_memory);

    // unsafe { device.bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0) }
    //     .expect("Error on bind buffer memory");


    // unsafe {
    //     let mut pointer: *mut std::ffi::c_void = std::ptr::null_mut();
    //     let mut ref1 = &mut pointer;
    //     device
    //         .map_memory(
    //             vertex_buffer_memory,
    //             256,
    //             vk::WHOLE_SIZE,
    //             None,
    //             ref1,
    //         )
    //         .expect("failed to map 333memory.");

    // }






let mut frame = 0;
    #[allow(clippy::collapsible_match, clippy::single_match)]
    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(StartCause::Init) => {
            *control_flow = ControlFlow::Poll;
        }
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            _ => (),
        },
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(keycode),
                state,
                ..
            }) => match (keycode, state) {
                (VirtualKeyCode::Escape, ElementState::Released) => {
                    *control_flow = ControlFlow::Exit
                }
                _ => (),
            },
            _ => (),
        },
        Event::MainEventsCleared => {
            unsafe {
                device
                    .wait_for_fences(&[in_flight_fences[frame]], true, u64::MAX)
                    .unwrap();
            }

            let image_index = unsafe {
                device.acquire_next_image_khr(
                    swapchain,
                    u64::MAX,
                    image_available_semaphores[frame],
                    vk::Fence::null(),
                )
            }
            .unwrap();

            let image_in_flight = images_in_flight[image_index as usize];
            if !image_in_flight.is_null() {
                unsafe { device.wait_for_fences(&[image_in_flight], true, u64::MAX) }.unwrap();
            }
            images_in_flight[image_index as usize] = in_flight_fences[frame];

            let wait_semaphores = vec![image_available_semaphores[frame]];
            let command_buffers = vec![cmd_bufs[image_index as usize]];
            let signal_semaphores = vec![render_finished_semaphores[frame]];
            let submit_info = vk::SubmitInfoBuilder::new()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            unsafe {
                let in_flight_fence = in_flight_fences[frame];
                device.reset_fences(&[in_flight_fence]).unwrap();
                device
                    .queue_submit(queue, &[submit_info], in_flight_fence)
                    .unwrap()
            }

            let swapchains = vec![swapchain];
            let image_indices = vec![image_index];
            let present_info = vk::PresentInfoKHRBuilder::new()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            unsafe { device.queue_present_khr(queue, &present_info) }.unwrap();

            frame = (frame + 1) % FRAMES_IN_FLIGHT;
        }
        Event::LoopDestroyed => unsafe {
            device.device_wait_idle().unwrap();

            for &semaphore in image_available_semaphores
                .iter()
                .chain(render_finished_semaphores.iter())
            {
                device.destroy_semaphore(semaphore, None);
            }

            for &fence in &in_flight_fences {
                device.destroy_fence(fence, None);
            }

            device.destroy_command_pool(command_pool, None);

            for &framebuffer in &swapchain_framebuffers {
                device.destroy_framebuffer(framebuffer, None);
            }

            device.destroy_pipeline(pipeline, None);

            device.destroy_render_pass(render_pass, None);

            device.destroy_pipeline_layout(pipeline_layout, None);

            device.destroy_shader_module(shader_vert, None);
            device.destroy_shader_module(shader_frag, None);

            for &image_view in &swapchain_image_views {
                device.destroy_image_view(image_view, None);
            }

            device.destroy_swapchain_khr(swapchain, None);

            device.destroy_device(None);

            instance.destroy_surface_khr(surface, None);

            if !messenger.is_null() {
                instance.destroy_debug_utils_messenger_ext(messenger, None);
            }

            instance.destroy_instance(None);

            println!("Exited cleanly");
        },
        _ => (),
    })
}


// this is a hacked together, partially complete attempt
// to separate out swapchain etc creation in preparation for the
// recreate_swapchain fn to be called when e.g. window is resized.
fn create_swapchain_etc(
    surface: & erupt::extensions::khr_surface::SurfaceKHR,
    format: vk::SurfaceFormatKHR,
    image_count: u32,
    surface_caps: erupt::extensions::khr_surface::SurfaceCapabilitiesKHR,
    present_mode: erupt::extensions::khr_surface::PresentModeKHR,
    device: & DeviceLoader,
    ) -> (
        erupt::extensions::khr_swapchain::SwapchainKHR,
        erupt::SmallVec<erupt::vk::Image>,
        Vec<erupt::vk::ImageView>
    ) {
    let swapchain_info = vk::SwapchainCreateInfoKHRBuilder::new()
        .surface(*surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(surface_caps.current_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain = unsafe { device.create_swapchain_khr(&swapchain_info, None) }.unwrap();
    let swapchain_images = unsafe { device.get_swapchain_images_khr(swapchain, None) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Image_views
    let swapchain_image_views: Vec<_> = swapchain_images
        .iter()
        .map(|swapchain_image| {
            let image_view_info = vk::ImageViewCreateInfoBuilder::new()
                .image(*swapchain_image)
                .view_type(vk::ImageViewType::_2D)
                .format(format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    *vk::ImageSubresourceRangeBuilder::new()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        // .build(),
                );
            unsafe { device.create_image_view(&image_view_info, None) }.unwrap()
        })
        .collect();

    (swapchain, swapchain_images, swapchain_image_views)
}


fn recreate_swapchain(device: & DeviceLoader) {
    unsafe { device.device_wait_idle(); }

    // create swap chain
    // create image views
    // create render pass
    // create graphics pipeline
    // create framebuffers
    // create command buffers
}



fn pretty_print(stuff: vk::PhysicalDeviceMemoryProperties) {
    println!("\n pretty_print physical_device_memory_properties: \n");
    for memory_type in stuff.memory_types {
        println!("memory type: {:?}", memory_type);
    }
    for heap in stuff.memory_heaps {
        println!("memory heap: {:?}", heap);
    }

}







    // let swapchain_info = vk::SwapchainCreateInfoKHRBuilder::new()
    //     .surface(surface)
    //     .min_image_count(image_count)
    //     .image_format(format.format)
    //     .image_color_space(format.color_space)
    //     .image_extent(surface_caps.current_extent)
    //     .image_array_layers(1)
    //     .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    //     .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
    //     .pre_transform(surface_caps.current_transform)
    //     .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
    //     .present_mode(present_mode)
    //     .clipped(true)
    //     .old_swapchain(vk::SwapchainKHR::null());

    // let swapchain = unsafe { device.create_swapchain_khr(&swapchain_info, None) }.unwrap();
    // let swapchain_images = unsafe { device.get_swapchain_images_khr(swapchain, None) }.unwrap();

    // // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Image_views
    // let swapchain_image_views: Vec<_> = swapchain_images
    //     .iter()
    //     .map(|swapchain_image| {
    //         let image_view_info = vk::ImageViewCreateInfoBuilder::new()
    //             .image(*swapchain_image)
    //             .view_type(vk::ImageViewType::_2D)
    //             .format(format.format)
    //             .components(vk::ComponentMapping {
    //                 r: vk::ComponentSwizzle::IDENTITY,
    //                 g: vk::ComponentSwizzle::IDENTITY,
    //                 b: vk::ComponentSwizzle::IDENTITY,
    //                 a: vk::ComponentSwizzle::IDENTITY,
    //             })
    //             .subresource_range(
    //                 vk::ImageSubresourceRangeBuilder::new()
    //                     .aspect_mask(vk::ImageAspectFlags::COLOR)
    //                     .base_mip_level(0)
    //                     .level_count(1)
    //                     .base_array_layer(0)
    //                     .layer_count(1)
    //                     .build(),
    //             );
    //         unsafe { device.create_image_view(&image_view_info, None) }.unwrap()
    //     })
    //     .collect();

    // ... let entry = 