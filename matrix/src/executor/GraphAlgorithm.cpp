//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/executor/GraphAlgorithm.h"

namespace matrix {
    int colorIndex = 0;

    int FindIndex(const std::vector<NodePtr> &nodes, const NodePtr &node) {
        for (int i = 0; i < nodes.size(); ++i) {
            if (node->id_ == nodes[i]->id_) {
                return i;
            }
        }
        return -1;
    }

    void FindBestPath(const std::vector<NodePtr> &nodes, const std::vector<int> &node_reward, std::vector<int>* path) {
        const int num_nodes = static_cast<int >(nodes.size());
        std::vector<int > best_reward(node_reward.size(), 0);
        std::vector<int> next_node(node_reward.size(), num_nodes);
        int best_solution = 0, best_start_node = 0;

        for (int i = num_nodes; i != 0; --i) {
            const int nid = i - 1;
            best_reward[nid] += node_reward[nid];
            if (best_reward[nid] > best_solution) {
                best_solution = best_reward[nid];
                best_start_node = nid;
            }
            for (const auto& e : nodes[nid]->inputs) {
                const int prev = FindIndex(nodes, e);
                if (prev == -1) continue;
                if (best_reward[nid] > best_reward[prev]) {
                    best_reward[prev] = best_reward[nid];
                    next_node[prev] = nid;
                }
            }
        }

        path->clear();
        int reward = 0;
        for (int nid = best_start_node; nid < num_nodes; nid = next_node[nid]) {
            path->push_back(nid); reward += node_reward[nid];
        }
    }


    void ColorLine(const std::vector<NodePtr> &nodes, std::map<size_t, int> &colors) {

        int n = nodes.size();
        if (n < 3) {
            for (auto &item : nodes) {
                colors[item->id_] = colorIndex++;
            }
        } else {

            auto checkInVectorFunc = [&](NodePtr current, NodePtr pre) -> bool {
                for (auto &it : pre->outputs) {
                    auto item = std::shared_ptr<Node>(it);
                    if ( item->id_ >= current->id_) {
                        return true;
                    }
                }
                return false;
            };

            auto checkSameColor = [&](NodePtr current, NodePtr pre) -> int {
                int color = colors[pre->id_];
                std::vector<size_t> nodeIds;
                for (auto &it : colors) {
                    if (color == it.second) {
                        nodeIds.push_back(it.first);
                    }
                }
                for (auto &item : current->inputs) {
                    for (size_t id : nodeIds) {
                        if (id == item->id_) {
                            return -1;
                        }
                    }
                }
                return color;
            };


            colors[nodes[0]->id_] = colorIndex++;
            for (int i = 1; i < n; ++i) {
                long preeps = -1;
                auto current = nodes[i];
                colors[current->id_] = -1;
                for (int j = i - 1; j >= 0 ; --j) {
                    auto pre = nodes[j];
                    if (checkInVectorFunc(current, pre)) {
                        if (j == 0 && colors[current->id_] == -1) {
                            colors[current->id_] = colorIndex++;
                        } else {
                            continue;
                        }
                    } else {
                        int  color = checkSameColor(current, pre);
                        if (color < 0){
                            if (j == 0 && colors[current->id_] == -1) {
                                colors[current->id_] = colorIndex++;
                            } else {
                                continue;
                            }
                        } else {
                            long currenteps = std::abs((long) pre->getMemorySize() - (long) current->getMemorySize());
                            if (preeps == -1) {
                                colors[current->id_] = color;
                            } else if (currenteps < preeps) {
                                colors[current->id_] = color;
                            }
                            preeps = currenteps;
                        }
                    }
                }
            }
        }

    }

    void ScatteredColorLine(const std::vector<NodePtr> &nodes, std::map<size_t, int> &colors) {

        size_t n = nodes.size();
        if (n <= 0) {
            return;
        }

        std::map<size_t , std::vector<int>> excludeColors;

        auto checkSameInput = [&](NodePtr &current, NodePtr &pre)->bool {
            for (auto &it : current->inputs) {
                for (auto &p : pre->inputs) {
                    if (it->id_ == p->id_) {
                        excludeColors[current->id_].push_back(colors[pre->id_]);
                        return true;
                    }
                }

            }
            return false;
        };

        auto checkLargeThenInput = [&](NodePtr &current, NodePtr &pre)->bool {
            for (auto &it : current->inputs) {
                if (it->id_ < pre->id_ && FindIndex(nodes, it) > 0) {
                    return false;
                }
            }
            return true;
        };



        auto checkSameColor = [&](NodePtr current, NodePtr pre) -> int {
            int color = colors[pre->id_];
            std::vector<size_t> nodeIds;
            for (auto &it : colors) {
                if (color == it.second) {
                    nodeIds.push_back(it.first);
                }
            }
            for (auto &item : current->inputs) {
                for (size_t id : nodeIds) {
                    if (id == item->id_) {
                        return -1;
                    }
                }
            }
            if (checkSameInput(current, pre)) {
                return -1;
            }
            if (excludeColors.count(current->id_)) {
                auto vec = excludeColors[current->id_];
                for(int col : vec) {
                    if (col == color) {
                        return -1;
                    }
                }
            }

            return color;
        };

        colors[nodes[0]->id_] = colorIndex++;
        for (int i = 1; i < n; ++i) {
            auto current = nodes[i];
            for (int j = i - 1; j >= 0 ; --j) {
                auto pre = nodes[j];
                if (checkLargeThenInput(current, pre) ) {
                    int flag = checkSameColor(current, pre);
                    if (flag > 0) {
                        colors[current->id_] = flag;
                        break;
                    } else {
                        if (j == 0) {
                            colors[current->id_] = colorIndex++;
                        } else {
                            continue;
                        }
                    }
                } else {
                    if (j == 0) {
                        colors[current->id_] = colorIndex++;
                    } else {
                        continue;
                    }
                }

            }
        }

    }

    void Remove(std::vector<NodePtr> &nodes, const std::vector<int> &path) {
        int index = 0;
        for (int id : path) {
            auto node = nodes[id - index];
            for(auto it=nodes.begin(); it != nodes.end();) {
                if (*it == node) {
                    it = nodes.erase(it);
                    index++;
                    break;
                } else {
                    ++it;
                }
            }
        }

    }

    GraphAlgorithm::GraphAlgorithm() {

    }

    void GraphAlgorithm::Coloring(Graph &graph, std::map<int, int> &graphColors) {
        graphColors.clear();
        colorIndex = 0;
        auto graphNodes = graph.GetGraphNodes();
        sort(graphNodes.begin(), graphNodes.end(), compare);
        std::vector<NodePtr> allUsefulNodeList; // 去除了正向传播所有输入， 还有不激活点

        for (auto &item : graphNodes) {
            if (item ->op == nullptr || item->isVariable) { // 常量node， 变量node， 不激活node不需要内存规划
                continue;
            }
            allUsefulNodeList.push_back(item);

        }

        std::vector<int > path;
        std::vector<std::vector<int >> paths;
        std::vector<int>node_reward(allUsefulNodeList.size());
        std::vector<NodePtr> last;

        last = allUsefulNodeList;
        while (last.size() > 0) {
            path.clear();
            node_reward.clear();
            for (auto &item : last) {
                node_reward.push_back(item->getMemorySize());
            }

            FindBestPath(last, node_reward, &path);
            std::vector<int> nodeIds;
            for (int id : path) {
                nodeIds.push_back(last[id]->id_);
//                LOG(INFO) << last[id]->name_ << "(id = " << last[id]->id_ << ", memorySize = " << last[id]->getMemorySize() << ")" << "---->";
            }
            paths.push_back(nodeIds);
            Remove(last, path);
        }


        // remove fetch node
        for (auto & it : fetchNodes) {
            graphColors[it->id_] = colorIndex++;
        }

        auto invector = [](std::vector<NodePtr> & nodes, int id) -> bool{
            for (auto &o : nodes) {
                if (o->id_ == id) {
                    return true;
                }
            }
            return false;
        };


        std::vector<int> lessThenTwo;
        std::map<size_t , int> colors;
        std::vector<NodePtr> line;
        for(auto &vec : paths) {
            if (vec.size() <= 2) {
                for (int i = 0; i < vec.size(); ++i) {
                    lessThenTwo.push_back(vec[i]);
                }
                continue;
            }
            for (int id : vec) {
                if (invector(fetchNodes, id)) {
                    continue;
                } else {
                    line.push_back(graph.GetNode(id));
                }
            }


            ColorLine(line, colors);
            line.clear();
        }

        // sort
        sort(lessThenTwo.begin(), lessThenTwo.end());

        line.clear();
        for (int id : lessThenTwo) {
            line.push_back(graph.GetNode(id));
        }

        ScatteredColorLine(line, colors);
        for(auto &it : colors) {
            graphColors.insert(it);
        }


        // todo::print
        for (auto &it : graphColors) {
//            LOG(INFO) << "the node(" << graph.GetNode(it.first)->name_ << "、id=" << it.first << ")" << "color is " << it.second;
        }
    }

    void GraphAlgorithm::Coloring(Graph &graph, std::map<int, int> &colors,
                                  const std::vector<NodePtr> &fetch) {
        this->fetchNodes = fetch;
        Coloring(graph, colors);
    }
}