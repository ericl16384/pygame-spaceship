import eric_lib as lib, numpy as np, pygame
import util


# Script

def main():
    pygame.display.set_mode(
        flags=pygame.RESIZABLE|pygame.FULLSCREEN
    )
    screen = pygame.display.get_surface()
    screen_rect = screen.get_rect()


    handle = lib.SimulationHandle(screen_rect.size)
    handle.add_category("SHIPS")

    ship = util.Ship(
        handle, (200, 200), 0, [
            util.Thruster((-20, -15), 0),
            util.Thruster((-20, 15), 0),
            # util.Thruster((25, -5), np.pi*-1/2),
            # util.Thruster((25, 5), np.pi*1/2)

            # util.Thruster((5, 15), np.pi),
            # util.Thruster((5, -15), np.pi)

            util.Thruster((25, -5), np.pi*-3/4),
            util.Thruster((25, 5), np.pi*3/4)
        ]
    )
    handle.categories["SHIPS"].append(ship)
    # ship.thrusters[2].thrust = 10


    clock = pygame.time.Clock()
    while True:
        # Display

        screen.blit(handle.render(), (0, 0))

        screen.blit(lib.render_text(
            f"FPS: {round(clock.get_fps())}",
            background=lib.WHITE
        ), (0, 0))

        pygame.display.flip()

        # print(f"position: {ship.pos}")
        # print(f"velocity: {ship.velocity}")
        # print(f"angle: {ship.angle}")
        # print(f"angular velocity: {ship.angular_velocity}")
        # input()


        # Update

        handle.update()

        # ship.pos = lib.vector((150, 100))
        # ship.angle = 0


        # Clock

        clock.tick(30)

if __name__ == "__main__":
    main()
    
