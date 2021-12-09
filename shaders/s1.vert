#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 inPosition;


layout(location = 0) out vec3 fragColor;

// vec2 positions[3] = vec2[](
//     vec2(0.0, -0.5),
//     vec2(0.5, 0.5),
//     vec2(-0.5, 0.5));



layout(set = 1, binding = 0) uniform cameraBuffer {
        mat4 model;
        mat4 view;
        mat4 proj;
    } cameraData;












vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0));









void main() {
    


    vec4 modelPos = cameraData.model * vec4(inPosition);

    vec4 viewPos = cameraData.view * modelPos;




    gl_Position = cameraData.proj * viewPos;







    // fragColor = colors[gl_VertexIndex % 3];
    fragColor = vec3(0.0, 1.0, 0.0);




}
