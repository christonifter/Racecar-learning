import numpy as np

class RaceTrack:
    def __init__(self, file_path:str):
        '''
        Initialize a RaceTrack object containing track data in a list of strings
        '''
        # racetrack map and dimensions
        with open(file_path, "r") as input_file:
            lines = input_file.readlines()
        shape_string = lines[0][0:(len(lines[0])-1)].split(',')
        self.track = lines[1:len(lines)] 
        self.ny = int(shape_string[0]) 
        self.nx = int(shape_string[1])
        self.pos_states = []
        self.pos_start = []
        for y in range(self.ny):
            for x in range(self.nx):
                if self.track[y][x] in 'S.':
                    self.pos_states.append([x,y])
                if self.track[y][x] in 'S':
                    self.pos_start.append([x,y])

        # racecar starting state
        start_pos = self.pos_start[np.random.choice(len(self.pos_start))]
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.v_x = 0
        self.v_y = 0


    def next_state(self, x, y, v_x, v_y, action, hardmode = False):
        '''
        Returns the reward and next state from performing (action) at state (x, y, v_x, v_y)

        Reward is: 
        0 if finish line is crossed without hitting a wall
        -1 otherwise

        Next state is:
        collision: last bresenham point before hitting a wall, 0 velocity
        finish line: finish line, 0 velocity, finishflag = True
        otherwise: mechanistic result of action
        # As long as the starting position is not in wall, the next state will not be in wall
        '''
        # accelerate velocity without exceeding speed of 5
        if abs(v_x + action[0]) < 6:
            v_x += action[0]
        if abs(v_y + action[1]) < 6:
            v_y += action[1]

        # check for collisions and finish line crossing
        points = self.bresenham(x, y, v_x, v_y)
        for i in range(len(points)):
            tracktype = self.track[points[i][1]][points[i][0]]
            # wall collision 
            if tracktype == '#':
                # move car to nearest starting line, velocity = 0
                if hardmode:
                    min_distance = 1000
                    for position in self.pos_start:
                        distance = ((points[i][0] - position[0])**2 + (points[i][1] - position[1])**2)**.5
                        if distance < min_distance:
                            min_distance = distance
                            start_pos = position
                    return [-1, start_pos[0], start_pos[1], 0, 0]
                # move car to last position before wall, velocity = 0 
                else:
                    return [-1, points[i-1][0], points[i-1][1], 0, 0]
            # cross finish line
            if tracktype == 'F':
                return [0, points[i][0], points[i][1], 0, 0]

        # no collisions
        return [-1, x + v_x, y + v_y, v_x, v_y]

    def bresenham(self, x0, y0, v_x, v_y):
        '''
        Returns the points crossed by the racecar based on Bresenham line algorithm
        Points are ordered from current point to next point
        '''
        reverse_flag = False
        if abs(v_x) < abs(v_y): # path travels farther along y than x, iterate across ys
            xi = 1
            if v_y < 0: # path travels south, start at end
                # reverse order
                reverse_flag = True
                x0 += v_x
                y0 += v_y
                v_x = -v_x
                v_y = -v_y
            if v_x < 0: # path travels west, shift x west when needed
                xi = -1
                v_x = -v_x
            D = 2*v_x - v_y
            out = []
            x = x0
            for y in range(y0, y0 + v_y +1): 
                out.append([x,y])
                if (D > 0): 
                    x += xi
                    D -= 2 * v_y
                D += 2 * v_x
        else: # path travels farther along x than y, iterate across xs
            yi = 1
            if v_x < 0: # path travels west, start at end
                reverse_flag = True
                # reverse order
                x0 += v_x
                y0 += v_y
                v_x = -v_x
                v_y = -v_y
            if v_y < 0: # path travels south, shift y south when needed
                yi = -1
                v_y = -v_y
            D = 2*v_y - v_x
            out = []
            y = y0
            for x in range(x0, x0 + v_x +1): 
                out.append([x,y])
                if (D > 0): 
                    y += yi
                    D -= 2 * v_x
                D += 2 * v_y
        if reverse_flag:
            out.reverse()
        return out

    def value_iteration(self, discount, epsilon = 1E-3, max_iterations = 1E3, hardmode = False):
        '''
        Estimates the value of all states based on Bellman's equation
        '''
        actions = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
        v_xs = list(np.arange(-5, 6))
        v_ys = list(np.arange(-5, 6))
        V = np.zeros((self.nx, self.ny, len(v_xs), len(v_ys)))
        Q = np.zeros((self.nx, self.ny, len(v_xs), len(v_ys), len(actions)))
        converge_flag = False
        iterations = 0
        while converge_flag == False:
            iterations += 1
            V_prev = V.copy()
            for position in self.pos_states:
                x = position[0]
                y = position[1]
                for i in range(len(v_xs)):
                    for j in range(len(v_ys)):
                        for k in range(len(actions)):
                            # V = 0 for all wall and finish line spaces. next_state will be never be in wall, and will be in finish line spaces only if 
                            # finish line is crossed.
                            (reward1, x_next, y_next, v_xnext, v_ynext) = self.next_state(x, y, v_xs[i], v_ys[j], actions[k], hardmode) # 80%
                            (reward2, x_next2, y_next2, v_xnext2, v_ynext2) = self.next_state(x, y, v_xs[i], v_ys[j], [0,0], hardmode) # 20%
                            expected_reward = 0.8 * reward1 + 0.2 * reward2
                            value_future = discount * (0.8 * V[x_next, y_next, v_xnext, v_ynext] + 0.2 * V[x_next2, y_next2, v_xnext2, v_ynext2])
                            Q[x, y, v_xs[i], v_ys[j], k] = expected_reward + value_future
                        V[x, y, v_xs[i], v_ys[j]] = max(Q[x, y, v_xs[i], v_ys[j], :])
            if np.abs(V-V_prev).max() < epsilon:
                converge_flag = True
            if iterations >= max_iterations:
                converge_flag = True
        return V, Q, iterations
    
    def q_learning(self, discount, learning_rate, decay = 0, T = 100, epsilon = 10, max_iterations = 1E5, hardmode = False, demo = 0):
        actions = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
        Q = np.random.uniform(-0.001, 0, size=(self.nx, self.ny, 11, 11, len(actions)))
        E = np.zeros((self.nx, self.ny, 11, 11, len(actions)))

        # iterate episodes
        converge_flag = False
        episodes = 0
        n_moves = []
        while converge_flag == False:
            episodes += 1
            start_pos = np.random.randint(len(self.pos_start))
            self.x = self.pos_start[start_pos][0]
            self.y = self.pos_start[start_pos][1]
            self.v_x = 0
            self.v_y = 0

            # iterate actions
            finished_flag = False
            moves = 0
            while finished_flag == False:
                moves += 1

                # choose action on softmax greedy choice
                P = np.exp(Q[self.x, self.y, self.v_x,  self.v_y, :]/T)/(np.exp(Q[self.x, self.y, self.v_x,  self.v_y, :]/T).sum())
                action_i = int(np.random.choice(len(actions), size=1, p=P))
                action = actions[action_i]
                if demo == 1 and episodes == max_iterations:
                    print(f'action softmax P = {P}, argmax P = {np.argmax(P)}, chosen action {action_i} ')

                # attempt action
                if np.random.rand(1) > .8:
                    action = [0, 0]
                    if demo == 2:
                        print('**acceleration failed**')
                (reward, x_next, y_next, v_xnext, v_ynext) = self.next_state(self.x, self.y, self.v_x, self.v_y, action, hardmode) # 80%

                # update Q
                dQ = learning_rate * (reward + discount * max(Q[x_next, y_next, v_xnext, v_ynext, :]) - Q[self.x, self.y, self.v_x,  self.v_y, action_i])
                # no eligibility trace
                if decay == 0:
                    Q[self.x, self.y, self.v_x,  self.v_y, action_i] = Q[self.x, self.y, self.v_x,  self.v_y, action_i] + dQ
                # eligibility trace
                else:
                    E = discount * decay * E
                    E[self.x, self.y, self.v_x,  self.v_y, action_i] = 1
                    Q = Q + dQ * E
                
                if demo == 2:
                    print(f'state [x, y, v_x, v_y] = {[self.x, self.y, self.v_x, self.v_y]}, attempted action = {actions[action_i]}, next_state [x, y, v_x, v_y] = {[x_next, y_next, v_xnext, v_ynext]} ')
                # move car
                self.x = x_next
                self.y = y_next
                self.v_x = v_xnext
                self.v_y = v_ynext
                if reward == 0:
                    finished_flag = True

            n_moves.append(moves)
#            print([moves, T, learning_rate])
#            print(np.var(np.array(n_moves[-100:len(n_moves)])))

            if episodes % 100 == 0:
#                print(np.var(np.array(n_moves[-100:len(n_moves)])))
                if np.var(np.array(n_moves[-100:len(n_moves)])) < epsilon:
                    converge_flag = True
            T = 1/(1/T+.001)
            learning_rate = 1/(1/learning_rate + .0001)
            if episodes >= max_iterations:
                converge_flag = True
        return Q, n_moves

    def sarsa(self, discount, learning_rate, decay = 0, T= 100, epsilon = 10, max_iterations = 1E5, hardmode = False, demo = 0):
        actions = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
        Q = np.random.uniform(-0.001, 0, size=(self.nx, self.ny, 11, 11, len(actions)))
        E = np.zeros((self.nx, self.ny, 11, 11, len(actions)))
#        T = 100 # temperature for simulated annealing

        # iterate episodes
        converge_flag = False
        episodes = 0
        n_moves = []
        while converge_flag == False:
            episodes += 1
            start_pos = np.random.randint(len(self.pos_start))
            self.x = self.pos_start[start_pos][0]
            self.y = self.pos_start[start_pos][1]
            self.v_x = 0
            self.v_y = 0

            # choose action on softmax greedy choice
            P = np.exp(Q[self.x, self.y, self.v_x,  self.v_y, :]/T)/(np.exp(Q[self.x, self.y, self.v_x,  self.v_y, :]/T).sum())
            action_i = int(np.random.choice(len(actions), size=1, p=P))
            if demo == 1:
                print(f'action softmax P = {P}, argmax P = {np.argmax(P)}, chosen action {action_i} ')

            # iterate actions
            finished_flag = False
            moves = 0
            while finished_flag == False:
                moves += 1
                action = actions[action_i]
                    
                # attempt action, observe reward and next state
                if np.random.rand(1) > .8:
                    action = [0, 0]
                    if demo == 2:
                        print('**acceleration failed**')
                (reward, x_next, y_next, v_xnext, v_ynext) = self.next_state(self.x, self.y, self.v_x, self.v_y, action, hardmode) # 80%

                # choose next action on softmax greedy choice
                P = np.exp(Q[x_next, y_next, v_xnext, v_ynext, :]/T)/(np.exp(Q[x_next, y_next, v_xnext, v_ynext, :]/T).sum())
                action_i_next = int(np.random.choice(len(actions), size=1, p=P))

                # update Q
                dQ = learning_rate * (reward + discount * Q[x_next, y_next, v_xnext, v_ynext, action_i_next] - Q[self.x, self.y, self.v_x,  self.v_y, action_i])
                # no eligibility trace
                if decay == 0:
                    Q[self.x, self.y, self.v_x,  self.v_y, action_i] = Q[self.x, self.y, self.v_x,  self.v_y, action_i] + dQ
                # eligibility trace
                else:
                    E = discount * decay * E
                    E[self.x, self.y, self.v_x,  self.v_y, action_i] = 1
                    Q = Q + dQ * E

                if demo == 2:
                    print(f'state [x, y, v_x, v_y] = {[self.x, self.y, self.v_x, self.v_y]}, attempted action = {actions[action_i]}, next_state [x, y, v_x, v_y] = {[x_next, y_next, v_xnext, v_ynext]} ')

                # move car
                self.x = x_next
                self.y = y_next
                self.v_x = v_xnext
                self.v_y = v_ynext
                action_i = action_i_next
                if reward == 0:
                    finished_flag = True

            n_moves.append(moves)
            if episodes % 100 == 0:
#                print([moves, T, learning_rate])
                if np.var(np.array(n_moves[-100:len(n_moves)])) < epsilon:
                    converge_flag = True
            T = 1/(1/T+.001)
            learning_rate = 1/(1/learning_rate + .0001)
            if episodes >= max_iterations:
                converge_flag = True
        return Q, n_moves
    
