import { useRef, useEffect } from 'react'
import { Renderer, Camera, Geometry, Program, Mesh } from 'ogl'

const vertex = `
attribute vec3 position;
attribute vec4 random;
uniform float uTime;
uniform float uSpread;
varying vec4 vRandom;
void main() {
    vRandom = random;
    vec3 pos = position;
    pos.x += sin(uTime * random.z * 0.4 + random.w * 6.28) * random.y * uSpread;
    pos.y += cos(uTime * random.w * 0.4 + random.z * 6.28) * random.y * uSpread;
    pos.z += sin(uTime * random.x * 0.3) * 0.5;
    gl_Position = vec4(pos, 1.0);
    gl_PointSize = mix(1.0, 3.0, random.x);
}
`

const fragment = `
precision highp float;
uniform vec3 uColor;
uniform float uAlpha;
varying vec4 vRandom;
void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    float alpha = uAlpha * smoothstep(0.5, 0.1, d) * (0.3 + vRandom.x * 0.7);
    gl_FragColor = vec4(uColor, alpha);
}
`

export default function Particles({
    className = '',
    count = 800,
    color = [0.83, 0.63, 0.09],
    spread = 1.2,
    speed = 0.3,
    alpha = 0.6,
}) {
    const containerRef = useRef(null)

    useEffect(() => {
        const el = containerRef.current
        if (!el) return

        const renderer = new Renderer({ alpha: true, antialias: true })
        const gl = renderer.gl
        gl.clearColor(0, 0, 0, 0)
        el.appendChild(gl.canvas)
        gl.canvas.style.width = '100%'
        gl.canvas.style.height = '100%'
        gl.canvas.style.display = 'block'

        const camera = new Camera(gl)
        camera.position.z = 3

        const resize = () => {
            const { width, height } = el.getBoundingClientRect()
            renderer.setSize(width, height)
            camera.perspective({ aspect: width / height })
        }
        resize()
        window.addEventListener('resize', resize)

        const positions = new Float32Array(count * 3)
        const randoms = new Float32Array(count * 4)
        for (let i = 0; i < count; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 3
            positions[i * 3 + 1] = (Math.random() - 0.5) * 3
            positions[i * 3 + 2] = (Math.random() - 0.5) * 2
            randoms[i * 4] = Math.random()
            randoms[i * 4 + 1] = Math.random()
            randoms[i * 4 + 2] = Math.random()
            randoms[i * 4 + 3] = Math.random()
        }

        const geometry = new Geometry(gl, {
            position: { size: 3, data: positions },
            random: { size: 4, data: randoms },
        })

        const program = new Program(gl, {
            vertex,
            fragment,
            uniforms: {
                uTime: { value: 0 },
                uSpread: { value: spread },
                uColor: { value: color },
                uAlpha: { value: alpha },
            },
            transparent: true,
            depthTest: false,
        })

        const mesh = new Mesh(gl, { mode: gl.POINTS, geometry, program })

        let raf
        const update = (t) => {
            raf = requestAnimationFrame(update)
            program.uniforms.uTime.value = t * 0.001 * speed
            renderer.render({ scene: mesh, camera })
        }
        raf = requestAnimationFrame(update)

        return () => {
            cancelAnimationFrame(raf)
            window.removeEventListener('resize', resize)
            if (gl.canvas.parentNode) gl.canvas.parentNode.removeChild(gl.canvas)
        }
    }, [count, color, spread, speed, alpha])

    return (
        <div
            ref={containerRef}
            className={className}
            style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none' }}
        />
    )
}
