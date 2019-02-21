# press ESC to exit the demo!
from pfilter import ParticleFilter, gaussian_noise, cauchy_noise, squared_error, independent_sample
import numpy as np

# testing only
from scipy.stats import norm, gamma, uniform
import skimage.draw
import cv2


img_size = 48


def blob(x):
    """Given an Nx3 matrix of blob positions and size, 
    create N img_size x img_size images, each with a blob drawn on 
    them given by the value in each row of x
    
    One row of x = [x,y,radius]."""
    y = np.zeros((x.shape[0], img_size, img_size))
    for i, particle in enumerate(x):
        rr, cc = skimage.draw.circle(
            particle[0], particle[1], particle[2], shape=(img_size, img_size)
        )
        y[i, rr, cc] = 1
    return y


#%%

# names (this is just for reference for the moment!)
columns = ["x", "y", "radius", "dx", "dy"]


# prior sampling function for each variable
# (assumes x and y are coordinates in the range 0-img_size)
prior_fn = independent_sample(
    [
        norm(loc=img_size / 2, scale=img_size / 2).rvs,
        norm(loc=img_size / 2, scale=img_size / 2).rvs,
        gamma(a=1, loc=0, scale=10).rvs,
        norm(loc=0, scale=0.5).rvs,
        norm(loc=0, scale=0.5).rvs,
    ]
)

# very simple linear dynamics: x += dx
def velocity(x):
    dt = 1.0
    xp = (
        x
        @ np.array(
            [
                [1, 0, 0, dt, 0],
                [0, 1, 0, 0, dt],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ).T
    )

    return xp


def test_filter():
    # create the filter
    pf = ParticleFilter(
        prior_fn=prior_fn,
        observe_fn=blob,
        n_particles=100,
        dynamics_fn=velocity,
        noise_fn=lambda x: cauchy_noise(x, sigmas=[0.05, 0.05, 0.01, 0.005, 0.005]),
        weight_fn=lambda x, y: squared_error(x, y, sigma=2),
        resample_proportion=0.2,
        column_names=columns,
    )

    # np.random.seed(2018)
    # start in centre, random radius
    s = np.random.uniform(2, 8)

    # random movement direction
    dx = np.random.uniform(-0.25, 0.25)
    dy = np.random.uniform(-0.25, 0.25)

    # appear at centre
    x = img_size // 2
    y = img_size // 2
    scale_factor = 20

    # create window
    cv2.namedWindow("samples", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("samples", scale_factor * img_size, scale_factor * img_size)

    for i in range(1000):
        # generate the actual image
        low_res_img = blob(np.array([[x, y, s]]))
        pf.update(low_res_img)

        # resize for drawing onto
        img = cv2.resize(
            np.squeeze(low_res_img), (0, 0), fx=scale_factor, fy=scale_factor
        )

        cv2.putText(
            img,
            "ESC to exit",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        color = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)

        x_hat, y_hat, s_hat, dx_hat, dy_hat = pf.mean_state

        # draw individual particles
        for particle in pf.original_particles:

            xa, ya, sa, _, _ = particle
            sa = np.clip(sa, 1, 100)
            cv2.circle(
                color,
                (int(ya * scale_factor), int(xa * scale_factor)),
                int(sa * scale_factor),
                (1, 0, 0),
                1,
            )

        # x,y exchange because of ordering between skimage and opencv
        cv2.circle(
            color,
            (int(y_hat * scale_factor), int(x_hat * scale_factor)),
            int(s_hat * scale_factor),
            (0, 1, 0),
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            color,
            (int(y_hat * scale_factor), int(x_hat * scale_factor)),
            (
                int(y_hat * scale_factor + 5 * dy_hat * scale_factor),
                int(x_hat * scale_factor + 5 * dx_hat * scale_factor),
            ),
            (0, 0, 1),
            lineType=cv2.LINE_AA,
        )

        cv2.imshow("samples", color)
        result = cv2.waitKey(20)
        # break on escape
        if result == 27:
            break
        x += dx
        y += dy

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_filter()
