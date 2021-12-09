


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


struct Peregrine<'a> {
    _entry: &'a erupt::EntryLoader,
    // in the actual struct used we'll move it instead of referencing it.
    // like in the ash tutorials.
    instance: vk::Instance,

    // surface_loader: self

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
struct FrameData {
    present_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
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
    println!("\nPeregrine Vulkan Workshop:\n");




    let frames : std::vec::Vec<FrameData> = vec!();



    // flush logs
    fs::remove_file("./logs/log_main.txt").unwrap();


    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Peregrine Ray-Trace")
        .build(&event_loop)
        .unwrap();

    let entry = EntryLoader::new().unwrap();



    // let peregrine = Peregrine{ _entry: & entry };


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



//  Created:  event_loop, window,  physical-device-properties, device-extensions, device-layers, present-mode, format, 
// queue-family, (logical)device-properties





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







// Now have created device.  Logical device.  Have created queue.  Have queried physical device surface capabilities,
// in order to coordinate the swapchain.  













    let (swapchain, swapchain_images, swapchain_image_views, swapchain_info) = create_swapchain_etc(
        &surface,
        format,
        image_count,
        surface_caps,
        present_mode,
        &device
    );



    // Now have created swapchain, swapchain-images, swapchain-image-views












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



    println!("\nVertices count {:?}\n", total_vertices_count);
    println!("\nIndices count: {:?}\n", indices.len());
    println!("\nDivides into:{:?}\n", indices.len() / total_vertices_count  );
    println!("\nIndices count / x: {:?}\n", indices.len() / 4 );


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


    // Now we've created some device buffer host coherent memory (slow) that stores some terrain data model.
    // Other types of memory available include gpu-only memory.  These can be loaded but not accessed by the cpu 
    // after the fact.  Something like that.







    let uniform_transform = UniformBufferObject {
        model: Matrix4::from_angle_z(Deg(90.0)),
        view: Matrix4::look_at_rh(
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ),
        proj: {
            let mut proj = cgmath::perspective(
                Deg(45.0),
                swapchain_info.image_extent.width as f32
                    / swapchain_info.image_extent.height as f32,
                0.1,
                10.0,
            );
            proj[1][1] = proj[1][1] * -1.0;
            proj
        },
    };


    // https://vkguide.dev/docs/chapter-4/descriptors/
    // Todo:  Make 4 descriptor sets as per the suggested in the link.








    let pool_sizes = [
        vk::DescriptorPoolSizeBuilder::new()
        ._type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1), // this count is the number of descriptors of this type to allocate
        vk::DescriptorPoolSizeBuilder::new()
        ._type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1),
    ];


    let x32 = vk::DescriptorPoolCreateFlags::all();

    println!("\nDescriptor Pool Flags created object: {:?}\n", x32);

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .flags(vk::DescriptorPoolCreateFlags::all()) // not clear how to pick indivulual flags with this builder function
        // is it a string that it makes (delimited by the bar |)?
        .max_sets(100) 
        .pool_sizes(&pool_sizes);

    let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None).unwrap() };


    let descriptor_set_layout_flags = vk::DescriptorSetLayoutCreateFlags::all();
         // Probably want to modify this.


    let sampler_x32_create_info = vk::SamplerCreateInfoBuilder::new()
        .flags(vk::SamplerCreateFlags::all())
        .mag_filter(vk::Filter(1))
        .min_filter(vk::Filter(1))
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .mip_lod_bias(0.0 as f32)
        .anisotropy_enable(true)
        .max_anisotropy(1.0 as f32)
        .compare_enable(false)
        .compare_op(vk::CompareOp::NEVER)
        .min_lod(1.0 as f32)
        .max_lod(2.0 as f32)
        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
        .unnormalized_coordinates(false);

        


    let sampler_x32 = device.create_sampler(&sampler_x32_create_info, None).unwrap();


    let model_view_proj_descriptor_set_layout_binding = vk::DescriptorSetLayoutBindingBuilder::new()
        .binding(1) // the number of this binding
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1) // There is only one descriptor in this set.
        //count?  so maybe the entire descriptor set has binding 1, and then you
        // can have different positions within that, to minutely address objects within the set.
        .stage_flags(vk::ShaderStageFlags::VERTEX) // I think this means it gets injected into the 
        // vertex stage as we see below in the shader stage flags bits.
        // Alternative may be to use the [...]Bits version, not sure the difference
        .immutable_samplers(&[]);





    // let descriptor_set_layout_binding_flags_create_info = 
    //     vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new()
    //         .binding_flags();  // DescriptorBindingFlags:  UPDATE_AFTER_BIND, PARTIALLY_BOUND (...)
    //                 // https://docs.rs/erupt/0.20.0+190/erupt/vk1_2/struct.DescriptorBindingFlags.html


    // let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
    //     .flags(descriptor_set_layout_flags)
    //     .bindings(& [descriptor_set_layout_binding]);


    // let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfoBuilder::new()
    //     .descriptor_pool(descriptor_pool)
    //     .set_layouts();


    let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
        .flags(descriptor_set_layout_flags)
        .bindings(&[model_view_proj_descriptor_set_layout_binding])
        .build_dangling();


    let descriptor_set_layout = unsafe { 
        device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None).unwrap() 
    };

    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(descriptor_pool)
        .set_layouts(& [descriptor_set_layout]);


    println!("\nuniform_transform: {:?}\n", uniform_transform);






    let descriptor_set_layout_support = vk::DescriptorSetLayoutSupportBuilder::new()
        .supported(true);

    println!("\ndescriptor_set_layout_support: {:?}\n", descriptor_set_layout_support);





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
            // We see here the usage pattern for the [...]Bits constructor.  It takes the 
            // string constant value and translates it.
            .module(shader_frag)
            .name(&entry_point),
    ];













    // let binding_description = VertexV3::get_binding_descriptions();
    // let attribute_description = VertexV3::get_attribute_descriptions();


    let vertex_input_binding_description = vk::VertexInputBindingDescriptionBuilder::new()
        .binding(0)
        .stride(std::mem::size_of::<VertexV3>() as u32,)
        .input_rate(vk::VertexInputRate::VERTEX);






    // So it looks like we've got binding 0 set on vertex_input, which I think bypasses
    // descriptor sets.



    let binding_descriptions = &[vertex_input_binding_description][..];




    let vertex_input_attribute_description = vk::VertexInputAttributeDescriptionBuilder::new()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .offset(offset_of!(VertexV3, pos) as u32);









    let attribute_descriptions = &[vertex_input_attribute_description][..];






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



/*The VkDescriptorSetLayoutBinding structure is defined as:

binding is the binding number of this entry and corresponds to a 
resource of the same binding number in the shader stages.

If there is just a binding number to this entry, a single resource per 
descriptor_set_layout_binding.  Given the name, that makes sense.

descriptorType is a VkDescriptorType specifying which type 
of resource descriptors are used for this binding.

Given the name.  Binding is to one specific resource.  Like a model-view-projection matrix.
Or something like a vertex buffer.

descriptorCount is the number of descriptors contained in the binding, 
accessed in a shader as an array, except if descriptorType is 
VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT in which case descriptorCount 
is the size in bytes of the inline uniform block. If descriptorCount is zero 
this binding entry is reserved and the resource must not be accessed from any 
stage via this binding within any pipeline using the set layout.

stageFlags member is a bitmask of VkShaderStageFlagBits specifying which 
pipeline shader stages can access a resource for this binding. 
VK_SHADER_STAGE_ALL is a shorthand specifying that all defined shader stages, 
ncluding any additional stages defined by extensions, can access the resource.

If a shader stage is not included in stageFlags,
 then a resource must not be accessed from that stage via this binding 
 within any pipeline using the set layout. Other than input attachments 
 which are limited to the fragment shader, there are no limitations on what 
 combinations of stages can use a descriptor binding, and in particular a 
 binding can be used by both graphics stages and the compute stage.
*/



    // This one is is for the uniform buffer object.
    // And, furthermore, the vertex buffer may go around another way besides descriptors,
    // I think may be known as push_constant.  This is seen in the vulkan-tutorial how
    // they introduce vertex_buffers
    let descriptor_set_layout_binding = vk::DescriptorSetLayoutBindingBuilder::new()
        .binding(0)  // Assign binding 0 to model-view-projection matrix. 
        .descriptor_type()
        .descriptor_count()
        .stage_flags()
        .immutable_samplers();


    let descriptor_set_layout = vk::DescriptorSetLayoutCreateInfoBuilder::new()
        .flags(vk::DescriptorSetLayoutCreateFlags::all())
        .bindings(& [descriptor_set_layout_binding]);


    let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
        .flags(
                vk::PiplelineLayoutCreateFlags::all()
            )
        .set_layouts(

            )
        .push_constant_ranges();





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



    // Here are created subpasses for the render-pass.  With new(), and then 
    // pipeline_bind_point(), and color_attachment().  This render-pass we've inherited is very sparse,
    // I imagine there are more complicated constructor chain functions to use in this.
    // Here it sets it to be a one-element vec.  
    // In a real program I think all of this is supposed to be configured at run-time, between frames.
    //  Dynamically generated rend-pass construction function.  For now we're just throwing the parts 
    // out on the floor.



    let subpasses = vec![vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)];




    // here are created dependencies for the render pass.
    // new(), and then src_subpass(), dst_subpass(), src_stage_mask(), 
    // src_access_mask(), dst_stage_mask, dst_access_mask,

    let dependencies = vec![vk::SubpassDependencyBuilder::new()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];









    // Now creating the render-pass.  With new(), attachments(), subpasses(), (dependencies)

    // "draw commands must be recorded within a render pass instance."
    // draw commands submitted to a command pool in a command buffer ?

    // syncronization commands introduce explicit execution dependencies, and memory dependencies, 
    // between two sets of operations.




    // Now prepare to make render-pass with render-pass-info construction.

    let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);


    // Create render-pass.
    let render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }.unwrap();



    // So now there is this other thing called the pipeline.  A pipeline is a containing structure for the
    // render-passes.  It binds these with buffers like vertex-input, viewport state, pipeline layout info, 
    // color-blend-state
    // rasterization state.




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



    // So there is the pipeline.   I think we've already created the swapchain stuff above, but that was just
    // the bare swapchain, the swapchain images and something.  Now we create swapchain-framebuffers.  Into this we put
    // attachments from image-view, width, height, the render-pass object.  





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


    // So now, with swapchain-framebuffers in hand, we can get ready the command-pool.



    // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers
    let command_pool_info =
        vk::CommandPoolCreateInfoBuilder::new().queue_family_index(queue_family);


    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }.unwrap();


    // All that needeed was putting in the queue-family.


    // Now command-buffers are created with the familiar pattern -- first creating an info object for the build.

    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swapchain_framebuffers.len() as _);
    let cmd_bufs = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }.unwrap();

    



    // Now we iterate through command-buffers and framebuffers in cmd_bufs.iter()



    for (&cmd_buf, &framebuffer) in cmd_bufs.iter().zip(swapchain_framebuffers.iter()) {




        let cmd_buf_begin_info = vk::CommandBufferBeginInfoBuilder::new();
        // Created an empty begin info object.



        unsafe { device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info) }.unwrap();
        // Atomic command to start the recording of the command buffer.


        let clear_values = vec![vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        // Set some clear values I guess for the swapchain images or frame-buffer images or something.



        let render_pass_begin_info = vk::RenderPassBeginInfoBuilder::new()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: surface_caps.current_extent,
            })
            .clear_values(&clear_values);


        // Render pass being recorded into the command buffer.  Using render-pass, framebuffer, render-area metadata, and clear values.



        // Here it is not actually being executed -- I think, it's being recorded:
        // "begin-render-pass", then bind a pipeline, and yes the pipeline holds the render-pass
        // and subpasses info, ... so this whole block is binding the pipeline to the cmd-buf
        // binding vertex buffers to the cmd-buf.
        // then its ends the device.cmd-render-pass.com
        // Here that is a one time thing, not clear under which conditions it will need to become dynamic.

        unsafe {
            device.cmd_begin_render_pass(
                cmd_buf,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, pipeline);

            // so now we are calling upon the logical device, its command cmd-begin-render-pass with the cmd-buf, render-pass begin info,
            // and SubpassContents type


            device.cmd_bind_vertex_buffers(cmd_buf, 0, &[vertex_buffer], &[256]);

            // Now we call upon logical device to bind vertex buffers, providing the cmd_buf, something 0, 
            // a slice with vertex-buffer, slice containing 256


            device.cmd_draw(cmd_buf, indices.len() as u32, 1, 0, 0);
            // Here device cmd-draw cmd-buf, indices.len() as u32   ... At this point we will need to verify the integrity
            // of our vertices and indices data.


            device.cmd_end_render_pass(cmd_buf);
            // Now we've ended the render pass.

            device.end_command_buffer(cmd_buf).unwrap();
            // Ended command buffer.
        }
    }

    // Do we need to do this in our event_loop for each frame?


    // Now comes some stuff with semaphores.

    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Rendering_and_presentation

    let semaphore_info = vk::SemaphoreCreateInfoBuilder::new();
    let image_available_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap())
        .collect();
    // We've created a semaphore each for the number of frames in flight.


    let render_finished_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap())
        .collect();
    // This may not actually be using these semaphores.  Looks like just building one for each frame in flight,
    // which is constant.



    // Now to fences.  Once we get complex render pass and pipeline structures going, there will be need to 
    // configure more particularly the fences.
    let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight_fences: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_fence(&fence_info, None) }.unwrap())
        .collect();
    let mut images_in_flight: Vec<_> = swapchain_images.iter().map(|_| vk::Fence::null()).collect();



    // And that's the end of that.  At least now we go into the event-loop.
    // render pass, event-loop.



    let mut frame = 0;
    // Before we start the event-loop, we set a frame counter variable.
    // We aren't yet 

    // After all the fences, semaphores, we have the command buffer submission.  
    // All this render-pass command-buffer recording is done statically before the render loop starts,
    // but I imagine that will go in a function to be able to produce render-pass structures dynamically. 




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

            // Now, in the main event loop flow, after main events cleared,
            // wait for device fences.


            unsafe {
                device
                    .wait_for_fences(&[in_flight_fences[frame]], true, u64::MAX)
                    .unwrap();
            }

            // we waited for fences.

            let image_index = unsafe {
                device.acquire_next_image_khr(
                    swapchain,
                    u64::MAX,
                    image_available_semaphores[frame],
                    vk::Fence::null(),
                )
            }
            .unwrap();

            // Now we define an image-index to device acquire-next-image-khr(), with the swapchain, image-available semaphores.

            let image_in_flight = images_in_flight[image_index as usize];

            // Now we get the image in flight with the index from images-in-flight vector.


            if !image_in_flight.is_null() {
                unsafe { device.wait_for_fences(&[image_in_flight], true, u64::MAX) }.unwrap();
            }

            // If the image-in-flght is no good, we call function on device to wait for fences.




            images_in_flight[image_index as usize] = in_flight_fences[frame];

            // We reset the in-flight-fences on indicated frame.


            let wait_semaphores = vec![image_available_semaphores[frame]];
            // Looks like we created a vec of wait semaphores we need to listen on?



            let command_buffers = vec![cmd_bufs[image_index as usize]];
            // We create new command buffers, here pulling them from our cmd_bufs array.



            let signal_semaphores = vec![render_finished_semaphores[frame]];
            //  Now we created some signal semaphores.


            let submit_info = vk::SubmitInfoBuilder::new()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            // Now something called submit-info


            unsafe {
                let in_flight_fence = in_flight_fences[frame];
                device.reset_fences(&[in_flight_fence]).unwrap();
                device
                    .queue_submit(queue, &[submit_info], in_flight_fence)
                    .unwrap()
            }

            let swapchains = vec![swapchain];
            // We only have one, but maybe there will be a few idk.  Mabye dynamically generated.


            let image_indices = vec![image_index];
            // Just giving one image to the queue-present-khr command, but it looks like you can give it a few.


            let present_info = vk::PresentInfoKHRBuilder::new()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            unsafe { device.queue_present_khr(queue, &present_info) }.unwrap();
            // This is finally giving device queue-present-khr command, which -- wow-- legit simply 
            // queues an image for display on the actual physical screen.  Params:
            // semaphores, swapchains, and image indices.

            // It looks like the render pass is not referenced at all in the event-loop, maybe in pipeline,
            // or maybe cmd-buf encompasses it.


            frame = (frame + 1) % FRAMES_IN_FLIGHT;
        }
        Event::LoopDestroyed => unsafe {

            // Need here to destroy everything Vulkan that needs destroying.  
            // I guess it's all destroyed when we close the window, but this enables
            // it's destruction if you wanted to continue the life of the program beyond this render-loop
            // without carrying on the memory baggage.
            // Some of these buffers I think have yet to receive their requisite destruction procedures.





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



// One function that would be a pain in the ass to re-inline above, so it's kind of out of pattern here

// Would be good to put together some testing unit testing for various pieces of in-line vs functionalized code.
// this is a hacked together, partially complete attempt
// to separate out swapchain etc creation in preparation for the
// recreate_swapchain fn to be called when e.g. window is resized.
fn create_swapchain_etc<'a>(
    surface: & erupt::extensions::khr_surface::SurfaceKHR,
    format: vk::SurfaceFormatKHR,
    image_count: u32,
    surface_caps: erupt::extensions::khr_surface::SurfaceCapabilitiesKHR,
    present_mode: erupt::extensions::khr_surface::PresentModeKHR,
    device: & DeviceLoader,
    ) -> (
        erupt::extensions::khr_swapchain::SwapchainKHR,
        erupt::SmallVec<erupt::vk::Image>,
        Vec<erupt::vk::ImageView>,
        vk::SwapchainCreateInfoKHRBuilder<'a>,
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


    println!("\nswapchain info: {:?}\n", swapchain_info);






    (swapchain, swapchain_images, swapchain_image_views, swapchain_info)
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


