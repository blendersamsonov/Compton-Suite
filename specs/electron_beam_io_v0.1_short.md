# I/O спецификация: гауссов электронный пучок v0.1 — краткая версия

Проект: Валидация Комптона  
Версия спецификации: `v0.1`  
Статус: краткий рабочий контракт

---

## 1. Scope

Документ задает минимальный I/O-контракт для одиночного гауссова электронного сгустка в расчетах Комптоновского рассеяния.

В v0.1 электронный пучок задается как 6D гауссово распределение с фокусом в точке взаимодействия.

---

## 2. Геометрическая конвенция

- Точка взаимодействия: $(x,y,z)=(0,0,0)$.
- Электроны распространяются вдоль $+z$.
- Лазер, если используется совместно со спецификацией лазера v0.1, распространяется вдоль $-z$.
- Фокус электронного пучка находится в точке взаимодействия.
- В фокусе: $\alpha_x=\alpha_y=0$.
- Корреляции $x$--$x'$ и $y$--$y'$ в v0.1 отсутствуют.

---

## 3. Input-параметры

| Поле | Обозначение | Единица | Описание |
|---|---:|---:|---|
| `bunch_charge_pC` | $Q$ | pC | Заряд одного электронного сгустка |
| `kinetic_energy_MeV` | $E_{\mathrm{kin}}$ | MeV | Средняя кинетическая энергия электронов |
| `rel_energy_spread_rms` | $\sigma_E/E$ | dimensionless | Относительный rms-разброс кинетической энергии |
| `sigma_x_rms_um` | $\sigma_{x0}$ | um | rms-размер пучка по $x$ в фокусе |
| `sigma_y_rms_um` | $\sigma_{y0}$ | um | rms-размер пучка по $y$ в фокусе |
| `emit_geom_x_um` | $\epsilon_x$ | um | Геометрический rms-эмиттанс по $x$ |
| `emit_geom_y_um` | $\epsilon_y$ | um | Геометрический rms-эмиттанс по $y$ |
| `bunch_duration_rms_ps` | $\sigma_t$ | ps | rms-длительность сгустка во времени |
| `propagation_direction` | — | — | В v0.1 фиксируется как `+z` |

Основная единица для заряда — `pC`, не `nC`.  
Конверсия: $1\ \mathrm{nC}=1000\ \mathrm{pC}$.

Геометрический эмиттанс в `um` означает:

$$
1\ \mathrm{um}=10^{-6}\ \mathrm{m\,rad}=1\ \mathrm{mm\,mrad}.
$$

---

## 4. Математическое определение

Поперечный профиль в фокусе:

$$
n(x,y) \propto
\exp\left(
-\frac{x^2}{2\sigma_{x0}^2}
-\frac{y^2}{2\sigma_{y0}^2}
\right).
$$

Временной профиль:

$$
n(t) \propto \exp\left(-\frac{t^2}{2\sigma_t^2}\right).
$$

Энергетическое распределение:

$$
f_E(\delta_E) \propto \exp\left(-\frac{\delta_E^2}{2\sigma_{\delta_E}^2}\right),
$$

где

$$
\sigma_{\delta_E}=\texttt{rel\_energy\_spread\_rms}.
$$

---

## 5. Эмиттанс и расходимость

В v0.1 используются геометрические rms-эмиттансы:

$$
\epsilon_x = \sqrt{\langle x^2 \rangle \langle x'^2 \rangle - \langle x x' \rangle^2},
$$

$$
\epsilon_y = \sqrt{\langle y^2 \rangle \langle y'^2 \rangle - \langle y y' \rangle^2}.
$$

В фокусе $\langle x x'\rangle=\langle y y'\rangle=0$, поэтому:

$$
\sigma_{x'}=\frac{\epsilon_x}{\sigma_{x0}},
$$

$$
\sigma_{y'}=\frac{\epsilon_y}{\sigma_{y0}}.
$$

Бета-функции в фокусе:

$$
\beta_x^*=\frac{\sigma_{x0}^2}{\epsilon_x},
$$

$$
\beta_y^*=\frac{\sigma_{y0}^2}{\epsilon_y}.
$$

---

## 6. Энергия и заряд

Средний лоренц-фактор:

$$
\gamma_0 = 1 + \frac{E_{\mathrm{kin}}}{m_ec^2},
$$

где

$$
m_ec^2 = 0.51099895\ \mathrm{MeV}.
$$

$$
\beta_0 = \sqrt{1 - \frac{1}{\gamma_0^2}}.
$$

Заряд:

$$
Q=\texttt{bunch\_charge\_pC}\times10^{-12}\ \mathrm{C}.
$$

Число электронов:

$$
N_e=\frac{Q}{e}.
$$

Пиковый ток для гауссова временного профиля:

$$
I_{\mathrm{peak}}=\frac{Q}{\sqrt{2\pi}\sigma_t}.
$$

---

## 7. Продольная длина и пиковая плотность

Пространственная rms-длина сгустка:

$$
\sigma_z=\beta_0 c\sigma_t.
$$

При `bunch_duration_rms_ps` в ps:

$$
\sigma_z[\mathrm{um}]\approx299.792458\,\beta_0\,\sigma_t[\mathrm{ps}].
$$

Пиковая плотность:

$$
n_0=\frac{N_e}{(2\pi)^{3/2}\sigma_x\sigma_y\sigma_z}.
$$

Если $\sigma_x$, $\sigma_y$, $\sigma_z$ заданы в cm, то $n_0$ получается в cm$^{-3}$.

---

## 8. Минимальный пример YAML

```yaml
electron_beam:
  model: gaussian_6d_waist
  version: "0.1"

  bunch_charge_pC: 100.0

  kinetic_energy_MeV: 200.0
  rel_energy_spread_rms: 0.001

  sigma_x_rms_um: 10.0
  sigma_y_rms_um: 10.0

  emit_geom_x_um: 0.05
  emit_geom_y_um: 0.05

  bunch_duration_rms_ps: 1.0

  propagation_direction: "+z"
```

---

## 9. Derived output

| Поле | Единица | Описание |
|---|---:|---|
| `bunch_charge_C` | C | Заряд сгустка в кулонах |
| `bunch_charge_nC` | nC | Заряд сгустка в нанокулонах |
| `num_electrons` | dimensionless | Число электронов $N_e$ |
| `gamma_mean` | dimensionless | Средний лоренц-фактор |
| `beta_mean` | dimensionless | Среднее $\beta=v/c$ |
| `sigma_E_kin_MeV` | MeV | rms-разброс кинетической энергии |
| `sigma_gamma_over_gamma` | dimensionless | Относительный rms-разброс по $\gamma$ |
| `sigma_z_rms_um` | um | rms-длина сгустка |
| `duration_fwhm_ps` | ps | FWHM-длительность |
| `divergence_rms_x_rad` | rad | rms-расходимость по $x$ |
| `divergence_rms_y_rad` | rad | rms-расходимость по $y$ |
| `beta_star_x_um` | um | $\beta_x^*$ в фокусе |
| `beta_star_y_um` | um | $\beta_y^*$ в фокусе |
| `emit_norm_x_um` | um | Нормированный rms-эмиттанс по $x$ |
| `emit_norm_y_um` | um | Нормированный rms-эмиттанс по $y$ |
| `peak_current_A` | A | Пиковый ток |
| `peak_density_cm3` | cm$^{-3}$ | Пиковая плотность электронов |

Нормированные эмиттансы:

$$
\epsilon_{n,x}=\beta_0\gamma_0\epsilon_x,
$$

$$
\epsilon_{n,y}=\beta_0\gamma_0\epsilon_y.
$$

---

## 10. Проверки валидности

```text
bunch_charge_pC > 0
kinetic_energy_MeV > 0
rel_energy_spread_rms >= 0
sigma_x_rms_um > 0
sigma_y_rms_um > 0
emit_geom_x_um > 0
emit_geom_y_um > 0
bunch_duration_rms_ps > 0
propagation_direction == "+z"
```

---

## 11. Не входит в v0.1

В v0.1 не входят:

- нормированные эмиттансы как input;
- Twiss-параметры как независимый input;
- смещение фокуса электронного пучка;
- поперечные и угловые смещения;
- дисперсия, energy chirp и $z$--$E$ корреляции;
- поперечно-продольные корреляции;
- slice emittance;
- негауссовы профили, halo и микробанчинг;
- спин и поляризация электронов;
- space charge, CSR, wakefields;
- эволюция пучка до и после точки взаимодействия.
