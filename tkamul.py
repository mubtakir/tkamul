"""
'طريقة رياضية جديدة لحل مسائل التكامل
'تم تطويره بواسطة: [باسل يحيى عبدالله/ العراق/ الموصل ]
تاريخ الإصدار: [20/4/2025]
"""

import torch
import matplotlib.pyplot as plt
import math

# -------------------------------
# تعريف فئة الحالة (CalculusState)
# -------------------------------
class CalculusState:
    def __init__(self, rep, D, V, tolerance):
        self.rep = rep.clone()       # المتجه المرجعي للحالة
        self.D = D.clone()           # معامل التفاضل
        self.V = V.clone()           # معامل التكامل
        self.tolerance = tolerance.clone()  # فرق مقبول لكل بعد
        self.usage = 1               # عدد مرات استخدام الحالة

    def forward(self, A):
        """
        حساب التفاضل والتكامل بناءً على الحالة:
            pred_dA = D * A
            pred_tA = V * A
        """
        pred_dA = self.D * A
        pred_tA = self.V * A
        return pred_dA, pred_tA

# --------------------------------------------------
# تعريف الخلية العصبية القائمة على الحالة (State‑Based)
# --------------------------------------------------
class StateBasedNeuroCalculusCell:
    def __init__(self, merge_threshold=0.8, learning_rate=0.3):
        self.states = []                 # قائمة الحالات
        self.merge_threshold = merge_threshold
        self.learning_rate = learning_rate

    def similarity(self, rep, A):
        """
        حساب درجة التشابه بين المتجه المرجعي (rep) والمدخل A باستخدام
        دالة تعتمد على المتوسط المطلق للفرق.
        """
        return torch.exp(-torch.mean(torch.abs(rep - A)))

    def process(self, A):
        """
        عند استلام المدخل A:
          - إذا لم توجد حالات، يتم إضافة حالة جديدة.
          - يتم البحث عن أفضل حالة (أو حالة تطابق تام إذا كان الفرق ضمن tolerance).
          - إذا كان أفضل تشابه أقل من عتبة الدمج، يتم إضافة حالة جديدة.
        يُرجع (الحالة المختارة، التفاضل المقدَّر، التكامل المقدَّر).
        """
        if len(self.states) == 0:
            self.add_state(A)
            best_state = self.states[-1]
        else:
            best_state = None
            best_sim = -1.0
            exact_match = False
            for state in self.states:
                if torch.all(torch.abs(state.rep - A) <= state.tolerance):
                    best_state = state
                    exact_match = True
                    break
                sim = self.similarity(state.rep, A)
                if sim > best_sim:
                    best_sim = sim
                    best_state = state
            if (not exact_match) and (best_sim < self.merge_threshold):
                self.add_state(A)
                best_state = self.states[-1]
        pred_dA, pred_tA = best_state.forward(A)
        return best_state, pred_dA, pred_tA

    def add_state(self, A):
        """
        إضافة حالة جديدة باستخدام المدخل A كمتجه مرجعي.
        تهيئة معاملات التفاضل والتكامل (D, V) بوحدة واحدة وتعيين tolerance مبدئي.
        """
        D_init = torch.ones_like(A)
        V_init = torch.ones_like(A)
        tol = torch.full_like(A, 0.5)
        new_state = CalculusState(rep=A, D=D_init, V=V_init, tolerance=tol)
        self.states.append(new_state)

    def adaptive_update(self, A, true_dA, true_tA):
        """
        تحديث الحالة المختارة بناءً على المدخل A:
          - تحديث المتجه المرجعي (rep) باستخدام متوسط مرجح.
          - تحديث معاملات D وV بحيث يكون:
                D * A ≈ true_dA  → target_D = true_dA / A
                V * A ≈ true_tA  → target_V = true_tA / A
          - زيادة usage وتحديث tolerance.
          - إجراء دمج تلقائي للحالات المتشابهة.
        """
        state, pred_dA, pred_tA = self.process(A)
        lr = self.learning_rate
        eps = 1e-6  # لتفادي القسمة على صفر

        state.rep = (1 - lr) * state.rep + lr * A
        target_D = true_dA / (A + eps)
        state.D = (1 - lr) * state.D + lr * target_D
        target_V = true_tA / (A + eps)
        state.V = (1 - lr) * state.V + lr * target_V

        state.usage += 1
        state.tolerance = state.tolerance * (1 + state.usage / 100.0)
        self.auto_merge()

    def auto_merge(self):
        """
        دمج الحالات المتشابهة تلقائيًا إذا كان التشابه بينها أعلى من عتبة الدمج.
        يتم تحديث المتجه المرجعي والمعاملات باستخدام متوسط مرجح قائم على usage.
        """
        merged_states = []
        for state in self.states:
            merged = False
            for m_state in merged_states:
                if self.similarity(state.rep, m_state.rep) >= self.merge_threshold:
                    total_usage = m_state.usage + state.usage
                    m_state.rep = (m_state.rep * m_state.usage + state.rep * state.usage) / total_usage
                    m_state.D = (m_state.D * m_state.usage + state.D * state.usage) / total_usage
                    m_state.V = (m_state.V * m_state.usage + state.V * state.usage) / total_usage
                    m_state.tolerance = torch.maximum(m_state.tolerance, state.tolerance)
                    m_state.usage = total_usage
                    merged = True
                    break
            if not merged:
                merged_states.append(state)
        self.states = merged_states

# -----------------------------------------
# تعريف دوال الاختبار مع بياناتها - دوال أكثر تعقيداً ونطاقات أوسع
# -----------------------------------------
test_functions = {
    'exp_wide': { # دالة أسية مع نطاق أوسع
        'f': lambda x: torch.exp(x),
        'f_prime': lambda x: torch.exp(x),
        'f_integral': lambda x: torch.exp(x) - 1,
        'domain': (-2.0, 4.0, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.2
    },
    'sin_wide': { # دالة جيبية مع نطاق أوسع
        'f': lambda x: torch.sin(x),
        'f_prime': lambda x: torch.cos(x),
        'f_integral': lambda x: 1 - torch.cos(x),  # بحيث f_integral(0)=0
        'domain': (0.0, 2*math.pi, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.2
    },
    'x^3_wide': { # دالة تكعيبية مع نطاق أوسع
        'f': lambda x: x**3,
        'f_prime': lambda x: 3*x**2,
        'f_integral': lambda x: x**4 / 4,
        'domain': (-2.0, 2.0, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.2
    },
    'cos_wide': { # دالة جيب تمام مع نطاق أوسع
        'f': lambda x: torch.cos(x),
        'f_prime': lambda x: -torch.sin(x),
        'f_integral': lambda x: torch.sin(x),  # نفترض نطاق [0, π/2]
        'domain': (0.0, math.pi, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.2
    },
    'piecewise_complex': { # دالة مقطعية أكثر تعقيداً
        'f': lambda x: torch.where(x < 0, torch.sin(x), torch.where(x < 1, x**2, torch.cos(x))),
        'f_prime': lambda x: torch.where(x < 0, torch.cos(x), torch.where(x < 1, 2*x, -torch.sin(x))),
        'f_integral': lambda x: torch.where(x < 0, -torch.cos(x) + 1, torch.where(x < 1, x**3/3 + 1 - 1, torch.sin(x) + (torch.tensor(1.0)/3.0) - torch.cos(torch.tensor(1.0)))),
        'domain': (-1.0, 2.0, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.3 # ضوضاء أعلى بسبب التعقيد
    },
    
    'abs_wide': { # دالة القيمة المطلقة مع نطاق أوسع
        'f': lambda x: torch.abs(x),
        'f_prime': lambda x: torch.sign(x),
        'f_integral': lambda x: 0.5 * x * torch.abs(x),
        'domain': (-4.0, 4.0, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.2
    },
    'sqrt_wide': { # دالة الجذر التربيعي مع نطاق أوسع
        'f': lambda x: torch.sqrt(x),
        'f_prime': lambda x: 0.5 * x**-0.5,
        'f_integral': lambda x: (2/3) * x**1.5,
        'domain': (0.01, 4.0, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.2
    },
    'sawtooth_wide': { # دالة سن المنشار مع نطاق أوسع
        'f': lambda x: torch.arctan(torch.tan(x)),
        'f_prime': lambda x: 1 / (1 + torch.tan(x)**2),
        'f_integral': lambda x:  x * torch.arctan(torch.tan(x)) - 0.5 * torch.log(1 + torch.tan(x)**2),
        'domain': (-math.pi + 0.1, math.pi - 0.1, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.3
    },
    'gaussian_wide': { # دالة جاوس مع نطاق أوسع
        'f': lambda x: torch.exp(-x**2),
        'f_prime': lambda x: -2*x*torch.exp(-x**2),
        'f_integral': lambda x: torch.sqrt(torch.tensor(math.pi)) * 0.5 * torch.erf(x),
        'domain': (-4.0, 4.0, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.2
    },
    'rational_wide': { # دالة كسرية مع نطاق أوسع
        'f': lambda x: x / (x**2 + 1),
        'f_prime': lambda x: (1 - x**2) / (x**2 + 1)**2,
        'f_integral': lambda x: 0.5 * torch.log(x**2 + 1),
        'domain': (-6.0, 6.0, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.2
    },
    'combined_complex': { # دالة مركبة أكثر تعقيداً
        'f': lambda x: torch.sin(2*x) + 0.5 * x**3 * torch.exp(-0.5*x) + torch.abs(x-1),
        'f_prime': lambda x: 2*torch.cos(2*x) + 1.5 * x**2 * torch.exp(-0.5*x) - 0.25 * x**3 * torch.exp(-0.5*x) + torch.sign(x-1),
        'f_integral': lambda x:  -0.5*torch.cos(2*x) + (-x**3 - 6*x**2 - 24*x - 48) * torch.exp(-0.5*x) -0.5 + 0.5 * (x-1) * torch.abs(x-1),
        'domain': (-2.0, 4.0, 200), # نطاق أوسع ونقاط أكثر
        'noise': 0.4 # ضوضاء أعلى بسبب التعقيد الشديد
    },
    'non_differentiable': { # دالة غير قابلة للاشتقاق في نقطة
        'f': lambda x: torch.where(x < 0, x**2, torch.sqrt(x)),
        'f_prime': lambda x: torch.where(x < 0, 2*x, 0.5 * x**-0.5), # مشتقة تقريبية، غير معرفة تماماً عند 0
        'f_integral': lambda x: torch.where(x < 0, x**3/3, (2/3) * x**1.5),
        'domain': (-2.0, 2.0, 200),
        'noise': 0.3 # ضوضاء أعلى بسبب عدم القابلية للاشتقاق
    }
}


# دالة لإضافة ضوضاء على التينسور
def add_noise(tensor, noise_level):
    noise = torch.randn_like(tensor) * noise_level * tensor.std()
    return tensor + noise

# دالة لحساب متوسط الخطأ المطلق (MAE)
def calculate_mae(predicted, true):
    return torch.mean(torch.abs(predicted - true))

# دالة لحساب متوسط مربع الخطأ (MSE)
def calculate_mse(predicted, true):
    return torch.mean((predicted - true)**2)

# -----------------------------------------
# التجربة الرئيسية: اختبار كل دالة وعرض النتائج مع التقييم الكمي
# -----------------------------------------
results = {} # قاموس لتخزين النتائج الكمية لكل دالة

for name, funcs in test_functions.items():
    print(f"\nاختبار الدالة: {name}")
    start, end, num_points = funcs['domain']
    x = torch.linspace(start, end, num_points)

    # حساب القيم الحقيقية للدالة والمشتقة والتكامل
    A = funcs['f'](x)
    true_dA = funcs['f_prime'](x)
    true_tA = funcs['f_integral'](x)

    # إضافة ضوضاء إذا كان مستوى الضوضاء محددًا
    noise_level = funcs.get('noise', 0.0)
    if noise_level > 0:
        A_noisy = add_noise(A, noise_level)
        true_dA_noisy = add_noise(true_dA, noise_level)
        true_tA_noisy = add_noise(true_tA, noise_level)
    else:
        A_noisy = A
        true_dA_noisy = true_dA
        true_tA_noisy = true_tA

    # تهيئة النموذج
    model = StateBasedNeuroCalculusCell(merge_threshold=0.8, learning_rate=0.3)
    epochs = 500
    loss_history = []

    for epoch in range(epochs):
        model.adaptive_update(A_noisy, true_dA_noisy, true_tA_noisy)
        _, pred_dA, pred_tA = model.process(A_noisy)
        loss_d = torch.mean(torch.abs(pred_dA - true_dA_noisy))
        loss_t = torch.mean(torch.abs(pred_tA - true_tA_noisy))
        total_loss = loss_d + loss_t
        loss_history.append(total_loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

    # الحصول على النتائج النهائية
    best_state, pred_dA, pred_tA = model.process(A_noisy)

    # حساب مقاييس التقييم الكمي
    mae_dA = calculate_mae(pred_dA, true_dA_noisy).item()
    mse_dA = calculate_mse(pred_dA, true_dA_noisy).item()
    mae_tA = calculate_mae(pred_tA, true_tA_noisy).item()
    mse_tA = calculate_mse(pred_tA, true_tA_noisy).item()

    results[name] = { # تخزين النتائج في القاموس
        'mae_dA': mae_dA,
        'mse_dA': mse_dA,
        'mae_tA': mae_tA,
        'mse_tA': mse_tA
    }

    print(f"  MAE التفاضل: {mae_dA:.4f}, MSE التفاضل: {mse_dA:.4f}")
    print(f"  MAE التكامل: {mae_tA:.4f}, MSE التكامل: {mse_tA:.4f}")

    # تحويل البيانات إلى numpy للرسم البياني
    x_np = x.numpy()
    true_dA_np = true_dA_noisy.numpy()
    pred_dA_np = pred_dA.detach().numpy()
    true_tA_np = true_tA_noisy.numpy()
    pred_tA_np = pred_tA.detach().numpy()

    # رسم مقارنة التفاضل
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"الدالة: {name}")

    plt.subplot(1, 2, 1)
    plt.plot(x_np, true_dA_np, label="المشتقة الحقيقية")
    plt.plot(x_np, pred_dA_np, '--', label="المشتقة المقدرة")
    plt.xlabel("x")
    plt.ylabel("dA")
    plt.title("مقارنة التفاضل")
    plt.legend()

    # رسم مقارنة التكامل
    plt.subplot(1, 2, 2)
    plt.plot(x_np, true_tA_np, label="التكامل الحقيقي")
    plt.plot(x_np, pred_tA_np, '--', label="التكامل المقدَّر")
    plt.xlabel("x")
    plt.ylabel("tA")
    plt.title("مقارنة التكامل")
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # رسم مسار الخسارة عبر الحقب
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label="الخسارة الإجمالية")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"مسار الخسارة للدالة {name}")
    plt.legend()
    plt.show()

    # عرض عدد الحالات النهائية ومعلومات الاستخدام
    print("عدد الحالات النهائية:", len(model.states))
    for i, state in enumerate(model.states):
        print(f"الحالة {i}: usage = {state.usage}")

# طباعة ملخص النتائج الكمية لجميع الدوال
print("\n-----------------------------------------")
print("ملخص النتائج الكمية لجميع الدوال:")
print("-----------------------------------------")
for name, metrics in results.items():
    print(f"الدالة: {name}")
    print(f"  MAE التفاضل: {metrics['mae_dA']:.4f}, MSE التفاضل: {metrics['mse_dA']:.4f}")
    print(f"  MAE التكامل: {metrics['mae_tA']:.4f}, MSE التكامل: {metrics['mse_tA']:.4f}")
    print("-----------------------------------------")