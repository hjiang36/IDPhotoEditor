#version 330 core

in vec2 TexCoords;
uniform sampler2D tex;
uniform vec2 texSize; // texture width, texture height
uniform sampler2D maskTexture;
uniform vec4 maskColor;
uniform vec3 brushInfo; // brush x, brush y, brush raius
uniform vec4 boundingBox; // left right, top, bottom

out vec4 outColor;

void main() {
    // Draw masked image as base layer.
    vec3 rgb = texture(tex, TexCoords).rgb;
    float alpha = texture(maskTexture, TexCoords).r;
    alpha = 1.0 - (1.0 - alpha) * maskColor.a;
    outColor = vec4(mix(maskColor.rgb, rgb, alpha), 1.0);

    // Draw brush circle on top of masked image.
    if (distance(TexCoords * texSize, brushInfo.xy * texSize) < brushInfo.z) {
        outColor = outColor * 0.5 + vec4(0.5, 0.2, 0.1, 0.5);
    }

    // Draw bounding box.
    if (boundingBox.x > -0.01) {
        vec2 xy = TexCoords * texSize;
        if (xy.x > boundingBox.x && xy.x < boundingBox.y && xy.y > boundingBox.z && xy.y < boundingBox.w) {
            outColor = outColor * 0.5 + vec4(0.1, 0.2, 0.5, 0.5);
        }
    }
}