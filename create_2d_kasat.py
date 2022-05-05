import matplotlib.pyplot as plt

from math import sqrt, floor

mix = lambda a, b, x: a*(1-x) + b*x
interpolant = lambda t: ((6*t - 15)*t + 10)*t*t*t
rng01 = lambda x: ((1103515245*x + 12345) % 2**32) / 2**32


def _gradient_noise(t):
    i = floor(t)
    f = t - i
    s0 = rng01(i)   * 2 - 1
    s1 = rng01(i + 1) * 2 - 1
    v0 = s0 * f;
    v1 = s1 * (f - 1);
    return mix(v0, v1, interpolant(f))


def _plot_noise(n, interp_npoints=100):
    xdata = [i/interp_npoints for i in range(n * interp_npoints)]
    gnoise = [_gradient_noise(x) for x in xdata]

    fig, ax = plt.subplots()
    ax.plot(xdata, gnoise, label='gradient noise')
    ax.set_xlabel('t')
    ax.set_ylabel('amplitude')
    ax.grid(linestyle=':')
    ax.legend(loc=1)

    x0, x1, y0, y1 = ax.axis()
    aspect = (y1 - y0) / (x1 - x0)

    for i in range(n + 1):
        dy = rng01(i) * 2 - 1  # gradient slope
        dx = 1
        norm = sqrt(dx**2 + (dy / aspect)**2)
        # norm *= 4  # 1/4 length
        vnx, vny = dx/norm, dy/norm
        x = (i-vnx/2, i+vnx/2)
        y = (-vny/2, vny/2)
        print(f"x{i}: ", x)
        ax.plot(x, y, 'r-')

    plt.show()


if __name__ == '__main__':
    _plot_noise(18)