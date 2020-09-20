$data = ['edges' => [], 'nodes' => []]; for($i=0;$i<25;$i++) { $data['nodes'][] = ['name' => 'Node '.$i, 'type' => 'type'.($i%7)]; $data['edges'][] = [$i, rand(0,24)]; } print_r(json_e
ncode($data));