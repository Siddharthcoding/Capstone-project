import os
import math
from typing import TypedDict, List
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────────────────
# DOMAIN METADATA
# ─────────────────────────────────────────────────────────

DOMAIN_NAME = "Physics Study Buddy"
DOMAIN_DESCRIPTION = (
    "A 24/7 intelligent study assistant for B.Tech students that explains "
    "Physics concepts faithfully and solves numericals step-by-step."
)

# ─────────────────────────────────────────────────────────
# DOCUMENTS
# ─────────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Newton's Laws of Motion",
        "text": """Newton's Three Laws of Motion form the foundation of classical mechanics.

First Law (Law of Inertia): An object remains at rest or in uniform motion in a straight line
unless acted upon by a net external force. Inertia is the property that causes this resistance.

Second Law (Law of Acceleration): The net force equals mass times acceleration: F = ma.
Force is measured in Newtons (N), where 1 N = 1 kg·m/s². F refers to the vector sum (net force).

Third Law (Action-Reaction): For every action there is an equal and opposite reaction. When
object A exerts a force on object B, object B simultaneously exerts an equal force in the
opposite direction on object A. These forces act on different bodies and do not cancel.

Key applications:
- Rocket propulsion: exhaust gases pushed backward, rocket pushed forward (3rd Law)
- Car braking: friction from road decelerates the vehicle (2nd Law)
- A book at rest on a table: gravity down, normal force up (1st Law equilibrium)

Weight: W = mg is the gravitational force on mass m where g = 9.8 m/s² near Earth's surface.
Mass is constant; weight varies with gravitational field strength.

Units: Force — Newton (N); Mass — kilogram (kg); Acceleration — m/s².""",
    },
    {
        "id": "doc_002",
        "topic": "Work, Energy, and Power",
        "text": """Work, Energy, and Power are interconnected scalar quantities in mechanics.

Work (W): Done when a force causes displacement. W = F · d · cos theta, where theta is the
angle between force and displacement. If force is perpendicular to displacement (theta = 90 deg),
work done is zero. Units: Joule (J) = N·m.

Kinetic Energy (KE): Energy of a moving object. KE = half * m * v^2, always positive.

Potential Energy (PE):
  Gravitational: U = mgh (m = mass, g = 9.8 m/s², h = height above reference)
  Elastic: U = half * k * x^2 (k = spring constant in N/m, x = deformation in m)

Work-Energy Theorem: Net work equals change in KE: W_net = delta_KE = half*m*v_f^2 - half*m*v_i^2

Conservation of Mechanical Energy: When only conservative forces act:
KE_initial + PE_initial = KE_final + PE_final.

Power (P): Rate of doing work. P = W/t = F·v. Units: Watt (W) = J/s.
1 horsepower = 746 W.

Efficiency: eta = (useful output energy / total input energy) x 100%.""",
    },
    {
        "id": "doc_003",
        "topic": "Circular Motion and Gravitation",
        "text": """Circular Motion: Object moves along a circular path. In uniform circular motion,
speed is constant but velocity direction changes — so acceleration exists.

Centripetal Acceleration: a_c = v^2/r, directed toward the center.
Centripetal Force: F_c = m*v^2/r = m*omega^2*r. Provided by tension, gravity, or friction.

Angular quantities:
  Angular velocity: omega = 2*pi*f = 2*pi/T (rad/s)
  Linear speed: v = omega * r
  Angular acceleration: alpha = delta_omega / delta_t (rad/s^2)

Universal Law of Gravitation: F = G * m1 * m2 / r^2
G = 6.674 x 10^-11 N·m²/kg², r = distance between centers.

Gravitational acceleration: g = G * M_E / R_E^2 ≈ 9.8 m/s²
M_E = 5.97 x 10^24 kg, R_E = 6.37 x 10^6 m.

Orbital velocity: v = sqrt(G*M/r)
Orbital period: T = 2*pi * sqrt(r^3 / (G*M))
Escape velocity from Earth: v_esc = sqrt(2*G*M_E/R_E) ≈ 11.2 km/s.""",
    },
    {
        "id": "doc_004",
        "topic": "Waves and Simple Harmonic Motion",
        "text": """Simple Harmonic Motion (SHM): Periodic motion where restoring force is
proportional to and opposite to displacement: F = -k*x.

Key SHM equations:
  Displacement: x(t) = A * cos(omega*t + phi)
  Velocity: v(t) = -A*omega * sin(omega*t + phi)
  Acceleration: a(t) = -omega^2 * x
  Angular frequency: omega = sqrt(k/m) for a spring-mass system
  Period: T = 2*pi/omega = 2*pi * sqrt(m/k)
  Frequency: f = 1/T

A = amplitude, phi = initial phase angle.
Simple Pendulum: T = 2*pi * sqrt(L/g) — period depends only on L and g, NOT mass.

Waves: Transfer energy without transferring matter.
  Transverse: oscillation perpendicular to propagation (light, string)
  Longitudinal: oscillation parallel to propagation (sound)

Wave equation: v = f * lambda (v = speed, f = frequency, lambda = wavelength).
Wave speed on a string: v = sqrt(T/mu), T = tension, mu = linear mass density.

Standing waves (string fixed at both ends):
  lambda_n = 2L/n,  f_n = n*v/(2L),  n = 1, 2, 3, ... (n=1 is fundamental).""",
    },
    {
        "id": "doc_005",
        "topic": "Thermodynamics — Laws and Processes",
        "text": """Thermodynamics studies heat, work, and energy transformations.

Zeroth Law: If A is in thermal equilibrium with B, and B with C, then A is in equilibrium
with C. This defines temperature.

First Law: delta_U = Q - W
delta_U = change in internal energy, Q = heat added (+in), W = work done BY system (+expand).

Processes:
  Isothermal (T=const): delta_U=0, Q=W; ideal gas: P*V=constant
  Adiabatic (Q=0): delta_U=-W; P*V^gamma=constant (gamma ~1.4 for diatomic gas)
  Isochoric (V=const): W=0, delta_U=Q
  Isobaric (P=const): W=P*delta_V

Second Law: Heat flows hot to cold spontaneously. No engine is 100% efficient. delta_S >= 0.
Third Law: Entropy of a perfect crystal at 0 K is zero.

Carnot Efficiency: eta = 1 - T_C/T_H (temperatures in Kelvin — maximum possible efficiency).

Ideal Gas Law: P*V = n*R*T
P (Pa), V (m^3), n (mol), R = 8.314 J/(mol·K), T (K).""",
    },
    {
        "id": "doc_006",
        "topic": "Electrostatics — Coulomb's Law and Electric Field",
        "text": """Electrostatics: behaviour of stationary electric charges.

Coulomb's Law: F = k * q1 * q2 / r^2
k = 9 x 10^9 N·m²/C², epsilon_0 = 8.85 x 10^-12 C²/(N·m²). Like charges repel, unlike attract.

Electric Field E: Force per unit positive test charge. E = k*Q/r^2 for a point charge Q.
Units: N/C = V/m. Field lines go from positive to negative charges.

Electric Potential V: Work per unit charge from infinity to a point. V = k*Q/r.
Relation: E = -dV/dr. Units: Volt (V) = J/C.

Potential energy of two charges: U = k * q1 * q2 / r

Gauss's Law: Phi_E = Q_enclosed / epsilon_0. Used for symmetric charge distributions.

Capacitor: C = Q/V.
Parallel-plate: C = epsilon_0 * A / d. Units: Farad (F). Energy stored: U = half*C*V^2.

Series capacitors: 1/C_total = 1/C1 + 1/C2 + ...
Parallel capacitors: C_total = C1 + C2 + ...""",
    },
    {
        "id": "doc_007",
        "topic": "Current Electricity — Ohm's Law and Circuits",
        "text": """Current Electricity: flow of charge through conductors.

Current I = Q/t. Units: Ampere (A) = C/s. Conventional current: + to - terminal.

Ohm's Law: V = I*R. V (Volt), I (Amp), R (Ohm). Valid for ohmic materials at constant T.

Resistance: R = rho * L / A. rho = resistivity (Ohm·m), L = length, A = cross-sectional area.
Temperature dependence: rho = rho_0 * [1 + alpha*(T - T_0)].

Series resistors: R_total = R1 + R2 + ... (same current through each)
Parallel resistors: 1/R_total = 1/R1 + 1/R2 + ... (same voltage across each)

Kirchhoff's Laws:
  KCL (Junction Rule): Sum of currents entering = sum of currents leaving any junction.
  KVL (Loop Rule): Sum of all voltage changes around any closed loop = 0.

Power dissipated: P = I*V = I^2*R = V^2/R. Units: Watt (W).

Real battery: EMF epsilon, internal resistance r. Terminal voltage: V_terminal = epsilon - I*r.
Wheatstone Bridge (balanced): R1/R2 = R3/R_x.
RC circuit charging: V_C(t) = epsilon*(1 - exp(-t/tau)), tau = R*C.""",
    },
    {
        "id": "doc_008",
        "topic": "Magnetic Fields and Electromagnetic Induction",
        "text": """Magnetism: interactions between moving charges and magnetic fields.

Magnetic force on a moving charge: F = q*v x B; magnitude F = q*v*B*sin(theta).
Units of B: Tesla (T). Force is always perpendicular to v and B — does no work.

Force on a current-carrying conductor: F = B*I*L*sin(theta).

Biot-Savart Law (long straight wire): B = mu_0 * I / (2*pi*r), mu_0 = 4*pi x 10^-7 T·m/A.
Ampere's Law: closed_loop_integral(B dL) = mu_0 * I_enclosed.
Solenoid: B = mu_0 * n * I (n = turns per unit length).

Faraday's Law: EMF = -d(Phi_B)/dt. Phi_B = B*A*cos(theta) in Weber (Wb = T·m^2).
Lenz's Law: Induced current opposes the change in flux (the negative sign).

Motional EMF: EMF = B*L*v (conductor length L moving at velocity v perpendicular to B).

Self Inductance L: EMF = -L * dI/dt; energy stored U = half*L*I^2. Units: Henry (H).
Mutual Inductance M: EMF_1 = -M * dI_2/dt.

Ideal Transformer: V_s/V_p = N_s/N_p = I_p/I_s (power conserved).""",
    },
    {
        "id": "doc_009",
        "topic": "Optics — Ray Optics and Wave Optics",
        "text": """Optics: behaviour and properties of light.

Reflection: angle of incidence = angle of reflection (both measured from normal).

Snell's Law: n1 * sin(theta_1) = n2 * sin(theta_2). n = c/v (refractive index).
c = 3 x 10^8 m/s. Light bends toward normal when entering denser medium (higher n).

Total Internal Reflection: occurs when light travels from dense to rare medium and angle
of incidence exceeds critical angle theta_c: sin(theta_c) = n2/n1.

Thin Lens Formula: 1/f = 1/v - 1/u. Magnification: m = v/u.
Lens Power: P = 1/f (f in meters). Units: Dioptre (D). Combined lenses: P_total = P1 + P2.

Mirror Formula: 1/f = 1/v + 1/u; f = R/2.

Young's Double Slit Experiment:
Fringe width: beta = lambda*D/d.
Bright fringes: path difference = n*lambda. Dark fringes: path difference = (2n-1)*lambda/2.
lambda = wavelength, D = screen distance, d = slit separation.

Single Slit Diffraction: dark fringes at a*sin(theta) = m*lambda (m = +/-1, +/-2, ...).""",
    },
    {
        "id": "doc_010",
        "topic": "Modern Physics — Quantum Theory and Photoelectric Effect",
        "text": """Modern Physics: phenomena beyond classical mechanics.

Planck's Hypothesis: energy emitted/absorbed in discrete quanta. E = h*f = h*c/lambda.
h = 6.626 x 10^-34 J·s (Planck's constant), f = frequency, lambda = wavelength.

Photoelectric Effect (Einstein 1905):
  Work function phi: minimum energy to eject an electron. h*f_0 = phi.
  Maximum KE of ejected electron: KE_max = h*f - phi = h*(f - f_0).
  Stopping potential V_0: e*V_0 = KE_max.

de Broglie Hypothesis: Every moving particle has a wavelength.
lambda = h/p = h/(m*v), where p = momentum.

Bohr's Model of Hydrogen:
  Energy of nth orbit: E_n = -13.6/n^2 eV (n = 1, 2, 3, ...)
  Radius of nth orbit: r_n = n^2 * a_0, a_0 = 0.529 Angstrom (Bohr radius)
  Photon emitted/absorbed: h*f = E_i - E_f.

Heisenberg Uncertainty Principle: delta_x * delta_p >= hbar/2, hbar = h/(2*pi).

Nuclear physics: A = Z + N (A = mass number, Z = protons, N = neutrons).
Binding energy: E_b = delta_m * c^2 (mass defect times c squared).
Radioactive decay: N(t) = N_0 * exp(-lambda*t). Half-life: T_half = 0.693/lambda.""",
    },
    {
        "id": "doc_011",
        "topic": "Special Relativity",
        "text": """Special Relativity (Einstein 1905): applies at speeds comparable to c.

Two Postulates:
  1. Laws of physics are identical in all inertial (non-accelerating) reference frames.
  2. The speed of light c = 3 x 10^8 m/s is the same for ALL observers regardless of
     the motion of the source or the observer.

Lorentz Factor: gamma = 1 / sqrt(1 - v^2/c^2), always >= 1.

Time Dilation: Moving clocks run slow. delta_t = gamma * delta_t_0.
  delta_t_0 = proper time (measured in the rest frame of the event).

Length Contraction: Moving objects are shorter along direction of motion. L = L_0 / gamma.
  L_0 = proper length (measured in the rest frame of the object).

Relativistic Momentum: p = gamma * m * v.

Mass-Energy Equivalence: rest energy E = m*c^2. Total energy: E = gamma*m*c^2.
Relativistic Kinetic Energy: KE = (gamma - 1)*m*c^2.

Relativistic Velocity Addition: u = (u_prime + v) / (1 + u_prime*v/c^2).
No object can reach or exceed the speed of light.

Invariant Spacetime Interval: s^2 = c^2*t^2 - x^2 is the same in all inertial frames.""",
    },
    {
        "id": "doc_012",
        "topic": "Fluid Mechanics and Surface Tension",
        "text": """Fluid Mechanics: behaviour of liquids and gases.

Pressure: P = F/A. Units: Pascal (Pa) = N/m^2.
Hydrostatic pressure at depth h: P = P_0 + rho*g*h (rho = fluid density, g = 9.8 m/s^2).

Pascal's Principle: Pressure applied to an enclosed fluid is transmitted equally in all directions.
Hydraulic Press: F1/A1 = F2/A2.

Archimedes' Principle: Buoyant force = weight of fluid displaced.
F_b = rho_fluid * V_displaced * g. Object floats if rho_object < rho_fluid.

Continuity Equation: A1*v1 = A2*v2 (volume flow rate Q = A*v is conserved for incompressible flow).

Bernoulli's Equation: P + half*rho*v^2 + rho*g*h = constant along a streamline.
Applications: aerofoil lift, Venturi meter, spray nozzle.

Viscosity (eta): internal friction of a fluid. Units: Pa·s.
Stokes' Law: drag force F = 6*pi*eta*r*v (sphere radius r, velocity v).
Terminal velocity: v_t = 2*r^2*(rho_sphere - rho_fluid)*g / (9*eta).

Surface Tension (sigma): force per unit length. Units: N/m.
Excess pressure inside a bubble: delta_P = 4*sigma/r (two surfaces).
Excess pressure inside a droplet: delta_P = 2*sigma/r.
Capillary rise: h = 2*sigma*cos(theta) / (rho*g*r), theta = contact angle.""",
    },
]

class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    student_name: str

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


def build_knowledge_base():
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(name="capstone_kb")

    if collection.count() == 0:
        print("Creating new collection and adding documents...")
        texts = [d["text"] for d in DOCUMENTS]
        ids = [d["id"] for d in DOCUMENTS]
        embeddings = embedder.encode(texts).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
        )
    else:
        print("Loaded existing collection")

    return embedder, collection


def make_memory_node():
    def memory_node(state: CapstoneState):
        msgs = list(state.get("messages", []))

        if not msgs or msgs[-1].get("role") != "user" or msgs[-1].get("content") != state["question"]:
            msgs.append({"role": "user", "content": state["question"]})

        msgs = msgs[-6:]

        student_name = state.get("student_name", "")
        q = state["question"].lower()
        for phrase in ["my name is ", "i am ", "i'm ", "call me "]:
            if phrase in q:
                after = state["question"][q.find(phrase) + len(phrase):].strip()
                if after:
                    candidate = after.split()[0].strip(".,!?")
                    if candidate.isalpha() and 2 <= len(candidate) <= 20:
                        student_name = candidate.title()
                        break

        return {"messages": msgs, "student_name": student_name}

    return memory_node


def make_router_node(llm):
    def router_node(state: CapstoneState):
        q = state["question"].strip().lower()

        if any(word in q for word in ["calculate", "compute", "evaluate", "find the value"]):
            return {"route": "tool"}

        if len(q.split()) < 5:
            return {"route": "memory_only"}

        return {"route": "retrieve"}

    return router_node


def make_retrieval_node(embedder, collection):
    def retrieval_node(state: CapstoneState):
        try:
            q_emb = embedder.encode([state["question"]]).tolist()
            results = collection.query(query_embeddings=q_emb, n_results=3)

            chunks = results["documents"][0] if results.get("documents") else []
            metadatas = results["metadatas"][0] if results.get("metadatas") else []

            if not chunks:
                return {"retrieved": "", "sources": []}

            topics = [m.get("topic", "Unknown") for m in metadatas]
            context = "\n\n---\n\n".join(
                f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
            )
            return {"retrieved": context, "sources": topics}

        except Exception as e:
            print("Retrieval error:", e)
            return {"retrieved": "", "sources": []}

    return retrieval_node


def skip_retrieval_node(state):
    return {"retrieved": "", "sources": []}


def tool_node(state):
    safe = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "g": 9.8,
        "G": 6.674e-11,
        "h": 6.626e-34,
        "c": 3e8,
        "k": 9e9,
        "R": 8.314,
        "NA": 6.022e23,
        "mu0": 4 * math.pi * 1e-7,
        "eps0": 8.85e-12,
        "sin": lambda x: math.sin(math.radians(x)),
        "cos": lambda x: math.cos(math.radians(x)),
        "tan": lambda x: math.tan(math.radians(x)),
        "asin": lambda x: math.degrees(math.asin(x)),
        "acos": lambda x: math.degrees(math.acos(x)),
        "atan": lambda x: math.degrees(math.atan(x)),
        "log": math.log10,
        "ln": math.log,
        "exp": math.exp,
        "abs": abs,
    }

    try:
        expr = state["question"].strip()
        for prefix in [
            "calculate ", "compute ", "evaluate ", "find ",
            "what is ", "solve ", "determine ", "what's "
        ]:
            if expr.lower().startswith(prefix):
                expr = expr[len(prefix):]
                break

        expr = expr.strip().rstrip("?.")
        result = eval(expr, safe, {})

        if isinstance(result, float):
            formatted = f"{result:.4e}" if (abs(result) < 1e-3 or abs(result) > 1e6) else f"{result:.6g}"
        else:
            formatted = str(result)

        tool_result = (
            f"Calculator result: {expr} = {formatted}\n"
            f"(Constants available: G={safe['G']}, h={safe['h']}, c={safe['c']}, "
            f"g={safe['g']}, k={safe['k']}, R={safe['R']}, NA={safe['NA']})"
        )

    except ZeroDivisionError:
        tool_result = "Calculator error: division by zero — please check your expression."
    except Exception as ex:
        tool_result = (
            "Calculator could not evaluate that expression. "
            "Please write it clearly, e.g. 'calculate sqrt(2*g*10)' or "
            "'find 9e9 * 1e-6 * 2e-6 / 0.1**2'. "
            f"(Error: {ex})"
        )

    return {"tool_result": tool_result}


def make_answer_node(llm):
    def generate_answer_node(state: CapstoneState):
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        student_name = state.get("student_name", "")

        greeting = f"the student named {student_name}" if student_name else "the student"

        context_parts = []
        if retrieved:
            context_parts.append(f"PHYSICS KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"CALCULATOR RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)

        if context:
            system_content = f"""You are a friendly and rigorous Physics Study Buddy for B.Tech students.
You are helping {greeting}.

STRICT RULES:
1. Explain using ONLY the formulas, definitions, and values in the context below.
   Do NOT add formulas or facts from your own training data.
2. If the answer is not in the context, say exactly:
   "I don't have that specific topic in my knowledge base. Please check your textbook or ask your professor."
3. Define every symbol in every formula (e.g., "where m = mass in kg").
4. For numerical problems, show working step-by-step with units at every step.
5. If a calculator result is provided, incorporate it naturally into the explanation.
6. Be encouraging and clear. Use the student's name if you know it.
7. Never guess physical constants — use only those stated in the context.

{context}"""
        else:
            system_content = (
                f"You are a friendly Physics Study Buddy helping {greeting}. "
                "Answer from the conversation history. "
                "If uncertain, tell the student to consult their textbook."
            )

        if eval_retries > 0:
            system_content += (
                "\n\nIMPORTANT: Your previous answer failed the faithfulness check. "
                "Use ONLY formulas and values explicitly present in the context above."
            )

        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            if msg["role"] == "user":
                lc_msgs.append(HumanMessage(content=msg["content"]))
            else:
                lc_msgs.append(AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    return generate_answer_node


def make_eval_node(llm):
    def eval_node(state: CapstoneState):
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:600]
        retries = state.get("eval_retries", 0)

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = (
            "Rate faithfulness: does this physics answer use ONLY the formulas and facts "
            "from the context? Reply with ONLY a number between 0.0 and 1.0.\n"
            "1.0 = fully faithful. 0.5 = some extra formulas added. 0.0 = mostly hallucinated.\n\n"
            f"Context: {context}\n\nAnswer: {answer[:400]}"
        )

        result = llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        return {"faithfulness": score, "eval_retries": retries + 1}

    return eval_node


def make_save_node():
    def save_node(state):
        msgs = list(state.get("messages", []))
        msgs.append({"role": "assistant", "content": state["answer"]})
        return {"messages": msgs}

    return save_node


def route_decision(state):
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    if route == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state):
    score = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


def build_graph(llm, embedder, collection):
    graph = StateGraph(CapstoneState)

    graph.add_node("memory", make_memory_node())
    graph.add_node("router", make_router_node(llm))
    graph.add_node("retrieve", make_retrieval_node(embedder, collection))
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("generate_answer", make_answer_node(llm))
    graph.add_node("eval", make_eval_node(llm))
    graph.add_node("save", make_save_node())

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )

    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("skip", "generate_answer")
    graph.add_edge("tool", "generate_answer")
    graph.add_edge("generate_answer", "eval")

    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "generate_answer", "save": "save"},
    )

    graph.add_edge("save", END)
    return graph.compile(checkpointer=MemorySaver())